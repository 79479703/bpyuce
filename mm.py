import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedFeatureWeightGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 改用Sigmoid激活函数
        )

    def forward(self, X):
        return self.net(X) * 2  # 将权重扩展到0-2范围增强特征影响

class OptimizedDynamicRFXGBoost:
    def __init__(self, n_estimators=200, base_lr=0.1, max_depth=6,
                 residual_steps=0, early_stopping_rounds=30,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.max_depth = max_depth
        self.residual_steps = residual_steps
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        self.models = []
        self.feature_net = None
        self.scaler_x = RobustScaler(quantile_range=(5, 95))
        self.best_val_loss = float('inf')
        self.no_improve = 0
        self.best_weights = None

    def _build_features(self, X, residual_history):
        enhanced_X = X.copy()
        for i in range(1, self.residual_steps + 1):
            if len(residual_history) >= i:
                residual = residual_history[-i][:X.shape[0]]
                enhanced_X = np.hstack([enhanced_X, residual.reshape(-1,1)])
            else:
                enhanced_X = np.hstack([enhanced_X, np.zeros((X.shape[0],1))])
        return enhanced_X

    def fit(self, X, y):
        # 数据预处理（仅对特征进行缩放）
        X = self.scaler_x.fit_transform(X)
        y = y.astype(np.float32)

        # 数据集划分
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        # 初始化训练参数
        residuals = y_train.copy()
        train_residuals = []
        self.feature_net = EnhancedFeatureWeightGenerator(X_train.shape[1] + self.residual_steps).to(self.device)
        optimizer = optim.AdamW(self.feature_net.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_estimators)

        best_models = []
        progress = tqdm(range(self.n_estimators))
        cumulative_val_pred = np.zeros_like(y_val)

        for epoch in progress:
            current_lr = self.base_lr * (1 + np.cos(np.pi * epoch / self.n_estimators)) / 2

            # 特征构建与加权
            enhanced_X = self._build_features(X_train, train_residuals)
            weights = self.feature_net(torch.FloatTensor(enhanced_X).to(self.device)).cpu().detach().numpy()
            weighted_X = enhanced_X * weights

            # 训练XGBoost
            dtrain = xgb.DMatrix(weighted_X, label=residuals)
            model = xgb.train({
                'max_depth': self.max_depth,
                'learning_rate': current_lr,
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist' if 'cuda' in self.device else 'hist',
                'reg_lambda': 1.0,
                'reg_alpha': 0.5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5
            }, dtrain, num_boost_round=50)  # 增加基模型复杂度

            # 更新残差
            residuals -= model.predict(dtrain) * current_lr
            train_residuals.append(residuals.copy())

            # 验证集预测
            val_enhanced = self._build_features(X_val, train_residuals)
            val_weights = self.feature_net(torch.FloatTensor(val_enhanced).to(self.device)).cpu().detach().numpy()
            val_pred = model.predict(xgb.DMatrix(val_enhanced * val_weights)) * current_lr
            cumulative_val_pred += val_pred

            # 早停机制
            current_loss = mean_squared_error(y_val, cumulative_val_pred)
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.no_improve = 0
                best_models = self.models.copy() + [(model, current_lr)]
                self.best_weights = self.feature_net.state_dict().copy()
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stopping_rounds:
                    print(f"\n早停于第 {epoch} 轮")
                    break

            self.models.append((model, current_lr))
            progress.set_description(f"当前损失: {current_loss:.4f}")

        if self.best_weights is not None:
            self.feature_net.load_state_dict(self.best_weights)
        self.models = best_models
        return self

    def predict(self, X):
        X = self.scaler_x.transform(X)
        total_pred = np.zeros(X.shape[0])
        pred_residuals = []

        for model, lr in self.models:
            enhanced_X = self._build_features(X, pred_residuals)
            weights = self.feature_net(torch.FloatTensor(enhanced_X).to(self.device)).cpu().detach().numpy()
            pred = model.predict(xgb.DMatrix(enhanced_X * weights)) * lr
            total_pred += pred
            pred_residuals.append(pred.copy())
            if len(pred_residuals) > self.residual_steps:
                pred_residuals.pop(0)

        return total_pred  # 直接返回原始尺度预测值

# 数据预处理
try:
    data = pd.read_excel('二胎.xlsx')

    # 确保使用与基础代码相同的特征列
    X = data.iloc[:, 1:11].values.astype(np.float32)  # 第2到第11列（共10列）
    y = data.iloc[:, 12].values.astype(np.float32)     # 第13列

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化优化后的模型
    model = OptimizedDynamicRFXGBoost(
        n_estimators=200,
        base_lr=0.03,         # 调整学习率与基础代码一致
        max_depth=3,          # 与基础代码相同树深
        residual_steps=0,     # 暂时禁用残差特征
        early_stopping_rounds=30
    )
    model.fit(X_train, y_train)

    # 评估函数
    def enhanced_evaluate(y_true, y_pred, label):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        print(f"\n=== {label}性能 ===")
        print(f'R²: {r2_score(y_true, y_pred):.4f}')
        print(f'MAE: {mean_absolute_error(y_true, y_pred):.2f}')
        print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}')

        # 可视化
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, c='#1f77b4', alpha=0.6, edgecolors='w')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')

        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.hist(residuals, bins=30, color='#2ca02c', edgecolor='k', alpha=0.7)
        plt.xlabel('残差')
        plt.ylabel('频数')
        plt.title('残差分布')

        plt.tight_layout()
        plt.show()

    enhanced_evaluate(y_test, model.predict(X_test), "测试集")
    enhanced_evaluate(y_train, model.predict(X_train), "训练集")

except Exception as e:
    print(f"运行时错误: {e}")