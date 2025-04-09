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

# 设置全局随机种子
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cpu')

# ----------------- 核心模型定义 -----------------
class MultiHeadFeatureInteraction(nn.Module):
    def __init__(self, input_dim, num_heads=10):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.pad_dim = (num_heads - (input_dim % num_heads)) % num_heads
        self.actual_dim = input_dim + self.pad_dim

        self.query = nn.Linear(input_dim, self.actual_dim)
        self.key = nn.Linear(input_dim, self.actual_dim)
        self.value = nn.Linear(input_dim, self.actual_dim)
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

class HybridXGBModel(xgb.XGBRegressor):
    def __init__(self, alpha=0.5, random_state=42, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.alpha = alpha  # 确保alpha能正确接收参数

    def _split_loss(self, y, left, right):
        base_loss = super()._split_loss(y, left, right)

        if self.alpha <= 0:
            return base_loss

        X_left, y_left = left
        X_right, y_right = right

        reg_loss_left = 0
        if len(y_left) > 1:
            try:
                beta_left, _, _, _ = np.linalg.lstsq(X_left, y_left, rcond=None)
                reg_loss_left = np.mean((y_left - X_left @ beta_left) ** 2)
            except:
                pass

        reg_loss_right = 0
        if len(y_right) > 1:
            try:
                beta_right, _, _, _ = np.linalg.lstsq(X_right, y_right, rcond=None)
                reg_loss_right = np.mean((y_right - X_right @ beta_right) ** 2)
            except:
                pass

        return (1 - self.alpha) * base_loss + self.alpha * (reg_loss_left + reg_loss_right)

class EnhancedFeatureProcessor(nn.Module):
    def __init__(self, input_dim, ablation_config):
        super().__init__()
        self.ablation = ablation_config
        self.input_dim = input_dim
        self.has_trainable_params = False

        # 特征权重生成器
        if self.ablation.get('use_nn_weights', True):
            self.weight_generator = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Sigmoid(),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )
            self.has_trainable_params = True
        else:
            self.weight_generator = nn.Identity()

        # 注意力机制
        if self.ablation.get('use_attention', True):
            self.attn = MultiHeadFeatureInteraction(
                input_dim=input_dim,
                num_heads=self.ablation.get('num_heads', 10)
            )
            self.has_trainable_params = True
        else:
            self.attn = nn.Identity()

        # 残差连接
        self.use_residual = self.ablation.get('use_residual', True)
        if self.use_residual:
            self.norm = nn.LayerNorm(input_dim)
            self.has_trainable_params = True
        else:
            self.norm = nn.Identity()

        # 特征变换
        if self.ablation.get('use_feature_transform', True):
            self.fc = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Linear(input_dim * 2, input_dim),
                nn.Sigmoid()
            )
            self.has_trainable_params = True
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        residual = x

        # 特征权重应用
        if self.ablation.get('use_nn_weights', True):
            weights = self.weight_generator(x)
            x = x * weights

        # 注意力机制
        if self.ablation.get('use_attention', True):
            x = self.attn(x)

        # 特征变换
        x = self.fc(x)

        # 残差连接
        if self.use_residual:
            x = x + residual
            x = self.norm(x)

        return x

class RobustDynamicBooster:
    def __init__(self, base_input_dim, ablation_config, n_estimators=100,
                 base_lr=0.1, residual_steps=10, xgboost_params=None):
        self.ablation = ablation_config
        self.base_input_dim = base_input_dim
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.xgboost_params = xgboost_params or {}

        # 残差特征处理
        if self.ablation.get('use_residual_features', True):
            self.residual_steps = residual_steps
            self.adjusted_input_dim = base_input_dim + residual_steps
        else:
            self.residual_steps = 0
            self.adjusted_input_dim = base_input_dim

        # 特征处理器
        self.feature_processor = EnhancedFeatureProcessor(
            input_dim=self.adjusted_input_dim,
            ablation_config=self.ablation
        ).to(device)

        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.models = []
        self.residuals = []
        self.best_loss = float('inf')
        self.no_improve = 0
        self.X_val = None
        self.y_val = None

    def _build_dynamic_features(self, X, residuals):
        if not self.ablation.get('use_residual_features', True):
            return X  # 直接返回原始特征

        augmented_X = X.copy()
        for i in range(1, self.residual_steps + 1):
            if len(residuals) >= i:
                residual = residuals[-i][:len(X)]
                augmented_X = np.hstack([augmented_X, residual.reshape(-1, 1)])
            else:
                augmented_X = np.hstack([augmented_X, np.zeros((len(X), 1))])
        return augmented_X

    def fit(self, X, y, val_ratio=0.2):
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )
        self.X_val = X_val_raw
        self.y_val = y_val

        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        params = list(self.feature_processor.parameters())
        if len(params) > 0:
            optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        else:
            optimizer = None
            scheduler = None

        residuals_history = []
        progress = tqdm(range(self.n_estimators))

        for epoch in progress:
            # 动态特征构建
            dyn_X = self._build_dynamic_features(X_train, residuals_history)
            dyn_tensor = torch.FloatTensor(dyn_X).to(device)

            # 特征处理（仅在启用NN权重时应用）
            if self.ablation.get('use_nn_weights', True):
                if optimizer is not None:
                    self.feature_processor.train()
                    processed_features = self.feature_processor(dyn_tensor)
                    weights = processed_features.cpu().detach().numpy()
                else:
                    with torch.no_grad():
                        weights = self.feature_processor(dyn_tensor).cpu().numpy()
                weighted_dyn_X = dyn_X * weights
            else:
                weighted_dyn_X = dyn_X  # 直接使用原始特征

            # 构建混合模型
            alpha_val = 0.3 if self.ablation.get('use_hybrid_loss', True) else 0.0
            model = HybridXGBModel(
                alpha=alpha_val,
                learning_rate=self.base_lr,
                **self.xgboost_params
            )
            model.fit(weighted_dyn_X, y_train)

            # 计算残差
            pred = model.predict(weighted_dyn_X)
            new_residual = (y_train - pred).reshape(-1, 1)
            residuals_history.append(new_residual)
            if len(residuals_history) > self.residual_steps:
                residuals_history.pop(0)

            # 验证步骤
            val_dyn = self._build_dynamic_features(X_val, residuals_history)
            val_tensor = torch.FloatTensor(val_dyn).to(device)

            if self.ablation.get('use_nn_weights', True):
                with torch.no_grad():
                    val_weights = self.feature_processor(val_tensor).cpu().numpy()
                val_weighted_dyn = val_dyn * val_weights
            else:
                val_weighted_dyn = val_dyn

            val_pred = model.predict(val_weighted_dyn)
            val_loss = mean_squared_error(y_val, val_pred)

            # 参数更新
            if optimizer is not None:
                optimizer.zero_grad()
                loss_tensor = torch.tensor(val_loss, requires_grad=True)
                loss_tensor.backward()
                optimizer.step()
                scheduler.step(val_loss)

            # 早停机制
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.no_improve = 0
                self.best_weights = self.feature_processor.state_dict().copy()
                self.best_models = self.models.copy() + [model]
            else:
                self.no_improve += 1
                if self.no_improve >= 100:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            progress.set_description(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        # 恢复最佳参数
        if len(params) > 0:
            self.feature_processor.load_state_dict(self.best_weights)
        self.models = self.best_models
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        total_pred = np.zeros(len(X_scaled))
        residuals = []

        for model in self.models:
            dyn_X = self._build_dynamic_features(X_scaled, residuals)
            if self.ablation.get('use_nn_weights', True):
                dyn_tensor = torch.FloatTensor(dyn_X).to(device)
                with torch.no_grad():
                    weights = self.feature_processor(dyn_tensor).cpu().numpy()
                weighted_dyn_X = dyn_X * weights
            else:
                weighted_dyn_X = dyn_X

            pred = model.predict(weighted_dyn_X)
            total_pred += pred
            residuals.append(pred.reshape(-1, 1))

            if len(residuals) > self.residual_steps:
                residuals.pop(0)

        return total_pred

# ----------------- 评估函数 -----------------
def evaluate(y_true, y_pred, label):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    print(f"\n{label}评估:")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, c='#1f77b4', edgecolors='w')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{label}预测对比')

    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, color='#2ca02c', edgecolor='k')
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title(f'{label}残差分布')

    plt.tight_layout()
    plt.show()

# ----------------- 消融实验 -----------------
if __name__ == "__main__":
    try:
        data = pd.read_excel('限制产奶量总上2.xlsx')
        X = data.iloc[:, 1:13].values.astype(np.float32)
        y = data.iloc[:, 13].values.astype(np.float32)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()

    # 消融实验配置
    ablation_configs = {
        '基础XGBoost': {
            'use_nn_weights': True,
            'use_attention': True,
            'use_residual':True,
            'use_feature_transform': True,
            'use_hybrid_loss': True,
            'use_residual_features': True

        }
    }

    xgboost_params = {
        'max_depth': 6,
        'n_estimators': 100,
        'reg_alpha': 0.8,
        'reg_lambda': 1.2,
        'tree_method': 'hist'
    }

    for exp_name, config in ablation_configs.items():
        print(f"\n=== 当前实验: {exp_name} ===")
        model = RobustDynamicBooster(
            base_input_dim=12,
            ablation_config=config,
            n_estimators=100,
            base_lr=0.1,
            residual_steps=10,
            xgboost_params=xgboost_params
        )
        model.fit(X, y)

        # 训练集评估
        y_pred_train = model.predict(X)
        evaluate(y, y_pred_train, f"{exp_name}-训练集")

        # 验证集评估
        if model.X_val is not None:
            y_pred_val = model.predict(model.X_val)
            evaluate(model.y_val, y_pred_val, f"{exp_name}-验证集")