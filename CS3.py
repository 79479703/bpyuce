import seaborn as sns  # 添加在文件最顶部的导入区域
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
        self._update_dim_for_attention()

        self.has_trainable_params = False
        if self.ablation.get('use_nn_weights', True):
            self.weight_generator = nn.Sequential(
                nn.Linear(self.actual_dim, 64),
                nn.Sigmoid(),
                nn.Linear(64, self.actual_dim),
                nn.Sigmoid()
            )
            self.has_trainable_params = True
        else:
            self.weight_generator = nn.Identity()

        if self.ablation.get('use_attention', True):
            self.attn = MultiHeadFeatureInteraction(
                input_dim=self.actual_dim,
                num_heads=self.ablation.get('num_heads', 7)
            )
            self.has_trainable_params = True
        else:
            self.attn = nn.Identity()

        self.use_residual = self.ablation.get('use_residual', True)
        if self.use_residual:
            self.norm = nn.LayerNorm(self.actual_dim)
            self.has_trainable_params = True
        else:
            self.norm = nn.Identity()

        if self.ablation.get('use_feature_transform', True):
            self.fc = nn.Sequential(
                nn.Linear(self.actual_dim, self.actual_dim * 2),
                nn.GELU(),
                nn.Linear(self.actual_dim * 2, self.actual_dim),
                nn.Sigmoid()
            )
            self.has_trainable_params = True
        else:
            self.fc = nn.Identity()


    def _update_dim_for_attention(self):
        if self.ablation.get('use_attention', True):
            temp_attn = MultiHeadFeatureInteraction(self.input_dim)
            self.actual_dim = temp_attn.actual_dim
        else:
            self.actual_dim = self.input_dim

    def forward(self, x):
        residual = x

        if x.size(1) < self.actual_dim:
            x = torch.nn.functional.pad(x, (0, self.actual_dim - x.size(1)))
        elif x.size(1) > self.actual_dim:
            x = x[:, :self.actual_dim]

        if self.ablation.get('use_nn_weights', True):
            weights = self.weight_generator(x)
            x = x * weights

        if self.ablation.get('use_attention', True):
            x = self.attn(x)

        x = self.fc(x)

        if self.use_residual:
            residual = residual[:, :self.actual_dim]
            x = x + residual
            x = self.norm(x)

        return x

# ----------------- 修正后的Booster类 -----------------
class RobustDynamicBooster:
    def __init__(self, base_input_dim, ablation_config, n_estimators=100,
                 base_lr=0.1, residual_steps=2, xgboost_params=None):
        self.ablation = ablation_config
        self.base_input_dim = base_input_dim
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.xgboost_params = xgboost_params or {}

        if self.ablation.get('use_residual_features', True):
            self.residual_steps = residual_steps
            self.adjusted_input_dim = base_input_dim + residual_steps
        else:
            self.residual_steps = 0
            self.adjusted_input_dim = base_input_dim

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
        augmented_X = X.copy()
        if self.ablation.get('use_residual_features', True):
            for i in range(1, self.residual_steps + 1):
                if len(residuals) >= i:
                    residual = residuals[-i][:len(X)]
                    augmented_X = np.hstack([augmented_X, residual.reshape(-1, 1)])
                else:
                    augmented_X = np.hstack([augmented_X, np.zeros((len(X), 1))])

        current_dim = augmented_X.shape[1]
        expected_dim = self.feature_processor.actual_dim
        if current_dim < expected_dim:
            augmented_X = np.pad(augmented_X, ((0,0), (0,expected_dim-current_dim)), mode='constant')
        elif current_dim > expected_dim:
            augmented_X = augmented_X[:, :expected_dim]
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
            dyn_X = self._build_dynamic_features(X_train, residuals_history)
            dyn_tensor = torch.FloatTensor(dyn_X).to(device)

            if optimizer is not None:
                self.feature_processor.train()
                processed_features = self.feature_processor(dyn_tensor)
                weights = processed_features.cpu().detach().numpy()
            else:
                with torch.no_grad():
                    weights = self.feature_processor(dyn_tensor).cpu().numpy()

            model = xgb.XGBRegressor(
                learning_rate=self.base_lr,
                **self.xgboost_params
            )
            model.fit(dyn_X * weights, y_train)

            pred = model.predict(dyn_X * weights)
            new_residual = (y_train - pred).reshape(-1, 1)
            residuals_history.append(new_residual)
            if len(residuals_history) > self.residual_steps:
                residuals_history.pop(0)

            val_dyn = self._build_dynamic_features(X_val, residuals_history)
            val_tensor = torch.FloatTensor(val_dyn).to(device)
            with torch.no_grad():
                val_weights = self.feature_processor(val_tensor).cpu().numpy()
            val_pred = model.predict(val_dyn * val_weights)
            val_loss = mean_squared_error(y_val, val_pred)

            if optimizer is not None:
                optimizer.zero_grad()
                loss_tensor = torch.tensor(val_loss, requires_grad=True)
                loss_tensor.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step(val_loss)

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
            dyn_tensor = torch.FloatTensor(dyn_X).to(device)
            with torch.no_grad():
                weights = self.feature_processor(dyn_tensor).cpu().numpy()
            pred = model.predict(dyn_X * weights)
            total_pred += pred
            residuals.append(pred.reshape(-1, 1))
            if len(residuals) > self.residual_steps:
                residuals.pop(0)

        return total_pred

# ----------------- 评估函数 -----------------
def evaluate(y_true, y_pred, label):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 打印评估指标
    print(f"\n{label}评估:")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")

    # 创建画布和子图
    plt.figure(figsize=(20, 6))

    # 子图1：预测值-真实值散点图
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, c='#1f77b4', edgecolors='w', s=40)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
    plt.xlabel('真实值', fontsize=12, fontweight='bold')
    plt.ylabel('预测值', fontsize=12, fontweight='bold')
    plt.title('预测值对比散点图', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 子图2：残差分布图
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, color='#2ca02c', edgecolor='k', alpha=0.7)
    sns.kdeplot(residuals, color='#d62728', lw=2, label='密度曲线')
    plt.xlabel('残差', fontsize=12, fontweight='bold')
    plt.ylabel('频数', fontsize=12, fontweight='bold')
    plt.title('残差分布分析', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 子图3：前100样本对比折线图
    plt.subplot(1, 3, 3)
    plt.plot(y_true[:100],
             color='#ff7f0e',
             linestyle='-',
             linewidth=2,
             marker='o',
             markersize=4,
             alpha=0.8,
             label='真实值')

    plt.plot(y_pred[:100],
             color='#1f77b4',
             linestyle='--',
             linewidth=2,
             marker='s',
             markersize=4,
             alpha=0.8,
             label='预测值')

    plt.xlabel('样本数量（个）', fontsize=12, fontweight='bold')
    plt.ylabel('数值', fontsize=12, fontweight='bold')
    plt.title('预测值与实际值对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right', frameon=True, facecolor='white')
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(0, 100, 10), fontsize=10)  # 设置x轴刻度
    plt.yticks(fontsize=10)

    # 调整布局
    plt.tight_layout(pad=3.0)
    plt.show()

# ----------------- 执行消融实验 -----------------
if __name__ == "__main__":
    try:
        data = pd.read_excel('限制产奶量总上2.xlsx')
        X = data.iloc[:, 1:13].values.astype(np.float32)
        y = data.iloc[:, 13].values.astype(np.float32)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()

    ablation_configs = {
        'attention_only': {
            'use_nn_weights':False,
            'use_attention':True,
            'use_residual': False,
            'use_feature_transform':True,
            'use_residual_features':True
        }
    }

    xgboost_params = {
        'max_depth': 7,
        'n_estimators': 52,
        'reg_alpha': 0.8,
        'reg_lambda': 1.2,
        'tree_method': 'hist'
    }

    for exp_name, config in ablation_configs.items():
        print(f"\n=== 当前实验: {exp_name} ===")
        model = RobustDynamicBooster(
            base_input_dim=12,
            ablation_config=config,
            n_estimators=108,
            base_lr=0.09,
            residual_steps=2,
            xgboost_params=xgboost_params
        )
        model.fit(X, y)

        y_pred_train = model.predict(X)
        evaluate(y, y_pred_train, f"{exp_name}-训练集")

        if model.X_val is not None:
            y_pred_val = model.predict(model.X_val)
            evaluate(model.y_val, y_pred_val, f"{exp_name}-验证集")