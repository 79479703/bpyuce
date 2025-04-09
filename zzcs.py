import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# 设置全局参数
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cpu')


# ----------------- 核心模型定义 -----------------
class MultiHeadFeatureInteraction(nn.Module):
    def __init__(self, input_dim, num_heads=7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.pad_dim = (num_heads - (input_dim % num_heads)) % num_heads
        self.actual_dim = input_dim + self.pad_dim

        self.query = nn.Linear(self.actual_dim, self.actual_dim)
        self.key = nn.Linear(self.actual_dim, self.actual_dim)
        self.value = nn.Linear(self.actual_dim, self.actual_dim)
        self.out = nn.Linear(self.actual_dim, input_dim)

    def forward(self, x):
        if self.pad_dim > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_dim))

        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out(context.squeeze(1))


class EnhancedFeatureProcessor(nn.Module):
    def __init__(self, input_dim, ablation_config):
        super().__init__()
        self.ablation = ablation_config
        self.input_dim = input_dim

        # 注意力维度计算
        if self.ablation['use_attention']:
            temp_attn = MultiHeadFeatureInteraction(input_dim)
            self.actual_dim = temp_attn.actual_dim
        else:
            self.actual_dim = input_dim

        # 特征变换模块
        if self.ablation['use_feature_transform']:
            self.transform = nn.Sequential(
                nn.Linear(self.actual_dim, self.actual_dim * 2),
                nn.GELU(),
                nn.Linear(self.actual_dim * 2, self.actual_dim),
                nn.Sigmoid()
            )
        else:
            self.transform = nn.Identity()

    def forward(self, x):
        # 特征维度调整
        if x.size(1) < self.actual_dim:
            x = torch.nn.functional.pad(x, (0, self.actual_dim - x.size(1)))
        elif x.size(1) > self.actual_dim:
            x = x[:, :self.actual_dim]

        # 注意力机制
        if self.ablation['use_attention']:
            x = MultiHeadFeatureInteraction(self.actual_dim)(x)

        # 特征变换
        x = self.transform(x)
        return x


# ----------------- 改进的Booster类 -----------------
class RobustDynamicBooster:
    def __init__(self, base_input_dim, ablation_config, n_estimators=100,
                 base_lr=0.1, residual_steps=4, xgboost_params=None):
        self.ablation = ablation_config
        self.base_input_dim = base_input_dim
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.xgboost_params = xgboost_params or {}
        self.residual_steps = residual_steps if self.ablation['use_residual_features'] else 0
        self.adjusted_input_dim = base_input_dim + self.residual_steps

        self.feature_processor = EnhancedFeatureProcessor(
            input_dim=self.adjusted_input_dim,
            ablation_config=self.ablation
        ).to(device)

        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.models = []
        self.residuals = []
        self.best_loss = float('inf')
        self.no_improve = 0

    def _build_dynamic_features(self, X, residuals):
        augmented_X = X.copy()
        if self.ablation['use_residual_features']:
            for i in range(1, self.residual_steps + 1):
                residual = residuals[-i][:len(X)] if len(residuals) >= i else np.zeros((len(X), 1))
                augmented_X = np.hstack([augmented_X, residual.reshape(-1, 1)])
        return augmented_X

    def fit(self, X, y, val_ratio=0.2):
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        residuals_history = []
        progress = tqdm(range(self.n_estimators))

        for epoch in progress:
            # 动态特征构建
            dyn_X = self._build_dynamic_features(X_train, residuals_history)
            dyn_tensor = torch.FloatTensor(dyn_X).to(device)

            # 特征处理
            with torch.no_grad():
                processed_features = self.feature_processor(dyn_tensor).cpu().numpy()

            # XGBoost训练
            model = xgb.XGBRegressor(
                learning_rate=self.base_lr,
                **self.xgboost_params
            )
            model.fit(dyn_X * processed_features, y_train)

            # 残差计算
            pred = model.predict(dyn_X * processed_features)
            new_residual = (y_train - pred).reshape(-1, 1)
            residuals_history.append(new_residual)
            if len(residuals_history) > self.residual_steps:
                residuals_history.pop(0)

            # 验证评估
            val_dyn = self._build_dynamic_features(X_val, residuals_history)
            val_pred = model.predict(val_dyn * self.feature_processor(torch.FloatTensor(val_dyn).cpu().numpy())
            val_loss = mean_squared_error(y_val, val_pred)

            # 模型保存逻辑
            if val_loss < self.best_loss:
                self.best_loss = val_loss
            self.best_models = self.models.copy() + [model]
            self.no_improve = 0
            else:
            self.no_improve += 1
            if self.no_improve >= 100:
                print(f"\nEarly stopping at epoch {epoch}")
            break

            progress.set_description(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        self.models = self.best_models
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        total_pred = np.zeros(len(X_scaled))
        residuals = []

        for model in self.models:
            dyn_X = self._build_dynamic_features(X_scaled, residuals)
            processed_features = self.feature_processor(torch.FloatTensor(dyn_X)).cpu().numpy()
            pred = model.predict(dyn_X * processed_features)
            total_pred += pred
            residuals.append(pred.reshape(-1, 1))
            if len(residuals) > self.residual_steps:
                residuals.pop(0)

        return total_pred


# ----------------- 评估函数（保持不变）-----------------
# 此处保留原始评估函数，与用户提供的代码相同

# ----------------- 执行消融实验 -----------------
if __name__ == "__main__":
    # 数据加载
    data = pd.read_excel('限制产奶量总上2.xlsx')
    X = data.iloc[:, 1:13].values.astype(np.float32)
    y = data.iloc[:, 13].values.astype(np.float32)

    # 消融实验配置
    ablation_config = {
        'use_attention': True,  # 保留注意力机制
        'use_feature_transform': True,  # 保留特征变换
        'use_residual_features': True,  # 保留残差特征
        'use_nn_weights': False,  # 移除神经网络权重
        'use_residual': False  # 移除残差连接
    }

    xgboost_params = {
        'max_depth': 7,
        'n_estimators': 52,
        'reg_alpha': 0.8,
        'reg_lambda': 1.2,
        'tree_method': 'hist'
    }

    # 模型训练和评估
    model = RobustDynamicBooster(
        base_input_dim=12,
        ablation_config=ablation_config,
        n_estimators=108,
        base_lr=0.09,
        residual_steps=4,
        xgboost_params=xgboost_params
    )
    model.fit(X, y)

    y_pred_train = model.predict(X)
    evaluate(y, y_pred_train, "消融实验-训练集")