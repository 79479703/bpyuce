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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cpu')


# ----------------- 模型定义 -----------------
class MultiHeadFeatureInteraction(nn.Module):
    def __init__(self, input_dim, num_heads=4):
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


class EnhancedFeatureProcessor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = MultiHeadFeatureInteraction(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.attn(x)
        x = self.norm(x + residual)
        return self.fc(x)


class HybridXGBModel(xgb.XGBRegressor):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _split_loss(self, y, left, right):
        base_loss = super()._split_loss(y, left, right)
        X_left, y_left = left
        X_right, y_right = right

        if len(y_left) > 1:
            beta_left = np.linalg.lstsq(X_left, y_left, rcond=None)[0]
            reg_loss_left = np.mean((y_left - X_left @ beta_left) ** 2)
        else:
            reg_loss_left = 0

        if len(y_right) > 1:
            beta_right = np.linalg.lstsq(X_right, y_right, rcond=None)[0]
            reg_loss_right = np.mean((y_right - X_right @ beta_right) ** 2)
        else:
            reg_loss_right = 0

        total_loss = (1 - self.alpha) * base_loss + self.alpha * (reg_loss_left + reg_loss_right)
        return total_loss


class RobustDynamicBooster:
    def __init__(self, base_input_dim, n_estimators=100, base_lr=0.08,
                 residual_steps=2, num_heads=2,
                 xgb_max_depth=6, xgb_n_estimators=80,
                 xgb_reg_alpha=0.8, xgb_reg_lambda=1.2):
        self.base_input_dim = base_input_dim
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.residual_steps = residual_steps
        self.num_heads = num_heads
        self.xgb_max_depth = xgb_max_depth
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda

        self.adjusted_input_dim = base_input_dim + residual_steps
        self.feature_processor = EnhancedFeatureProcessor(self.adjusted_input_dim).to(device)
        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.models = []
        self.residuals = []
        self.best_loss = float('inf')
        self.no_improve = 0
        self.X_val = None
        self.y_val = None

    # 修改_build_dynamic_features方法（位于RobustDynamicBooster类中）
    def _build_dynamic_features(self, X, residuals):
        augmented_X = X.copy()
        n_samples = X.shape[0]  # 获取当前样本数量

        for i in range(1, self.residual_steps + 1):
            if len(residuals) >= i:
                residual = residuals[-i]

                # 统一转换为二维数组
                if residual.ndim == 1:
                    residual = residual.reshape(-1, 1)

                # 确保长度匹配
                if residual.shape[0] < n_samples:
                    # 长度不足时用零填充
                    padded_residual = np.zeros((n_samples, 1))
                    padded_residual[:residual.shape[0]] = residual
                else:
                    # 截断到当前样本数量
                    padded_residual = residual[:n_samples]

                augmented_X = np.hstack([augmented_X, padded_residual])
            else:
                # 初始化时填充零
                augmented_X = np.hstack([augmented_X, np.zeros((n_samples, 1))])

        return augmented_X.astype(np.float32)

    def fit(self, X, y, val_ratio=0.2):
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )
        self.X_val = X_val_raw
        self.y_val = y_val

        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        optimizer = optim.AdamW(self.feature_processor.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        residuals_history = []
        progress = tqdm(range(self.n_estimators))

        for epoch in progress:
            dyn_X = self._build_dynamic_features(X_train, residuals_history)
            dyn_tensor = torch.FloatTensor(dyn_X).to(device)

            with torch.no_grad():
                weights = self.feature_processor(dyn_tensor).cpu().numpy()

            current_lr = self.base_lr * (0.5 * (1 + np.cos(np.pi * epoch / self.n_estimators)))

            model = HybridXGBModel(
                alpha=0.3,
                learning_rate=current_lr,
                max_depth=self.xgb_max_depth,
                n_estimators=self.xgb_n_estimators,
                reg_alpha=self.xgb_reg_alpha,
                reg_lambda=self.xgb_reg_lambda,
                tree_method='hist'
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

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.no_improve = 0
                self.best_weights = self.feature_processor.state_dict().copy()
                self.best_models = self.models.copy() + [model]
            else:
                self.no_improve += 1
                if self.no_improve >= 50:
                    print(f"\n早停于第 {epoch} 轮")
                    break

            scheduler.step(val_loss)
            progress.set_description(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

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


# ----------------- 遗传算法修正版 -----------------
class GeneticOptimizer:
    def __init__(self, param_ranges, pop_size=10, generations=5,
                 mutation_rate=0.1, tournament_size=3):
        self.param_ranges = param_ranges
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def create_individual(self):
        individual = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            if param in ['booster_n_estimators', 'xgb_n_estimators', 'xgb_max_depth']:
                individual[param] = int(np.random.randint(min_val, max_val + 1))
            else:
                individual[param] = float(np.random.uniform(min_val, max_val))
        return individual

    def evaluate_individual(self, individual, X, y):
        try:
            model = RobustDynamicBooster(
                base_input_dim=X.shape[1],
                n_estimators=int(individual['booster_n_estimators']),
                base_lr=float(individual['base_lr']),
                residual_steps=2,
                num_heads=4,
                xgb_max_depth=int(individual['xgb_max_depth']),
                xgb_n_estimators=int(individual['xgb_n_estimators']),
                xgb_reg_alpha=0.8,
                xgb_reg_lambda=1.2
            )
            model.fit(X, y)
            return model.best_loss
        except Exception as e:
            print(f"评估失败: {str(e)}")
            return float('inf')

    def select_parents(self, population, fitness):
        parents = []
        for _ in range(len(population)):
            indices = np.random.choice(len(population), self.tournament_size, replace=False)
            selected_fitness = [fitness[i] for i in indices]
            best_index = indices[np.argmin(selected_fitness)]
            parents.append(population[best_index])
        return parents

    def crossover(self, parent1, parent2):
        child = {}
        for param in self.param_ranges.keys():
            if np.random.rand() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child

    def mutate(self, individual):
        mutated = individual.copy()
        for param in self.param_ranges.keys():
            if np.random.rand() < self.mutation_rate:
                min_val, max_val = self.param_ranges[param]
                if param in ['booster_n_estimators', 'xgb_n_estimators', 'xgb_max_depth']:
                    mutated[param] = int(np.random.randint(min_val, max_val + 1))
                else:
                    mutated[param] = float(np.random.uniform(min_val, max_val))
        return mutated

    def optimize(self, X, y):
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_individual = None
        best_fitness = float('inf')

        for gen in range(self.generations):
            print(f"\nGeneration {gen + 1}/{self.generations}")
            fitness = []
            for idx, ind in enumerate(population):
                print(f"Evaluating individual {idx + 1}/{self.pop_size}")
                current_fitness = self.evaluate_individual(ind, X, y)
                fitness.append(current_fitness)
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_individual = ind.copy()
                    print(f"New best fitness: {best_fitness:.4f}")

            parents = self.select_parents(population, fitness)

            next_generation = []
            while len(next_generation) < self.pop_size:
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation

        return best_individual, best_fitness


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


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    try:
        data = pd.read_excel('二胎.xlsx')
        X = data.iloc[:, 1:11].values.astype(np.float32)
        y = data.iloc[:, 12].values.astype(np.float32)

        param_ranges = {
            'booster_n_estimators': (50, 200),
            'base_lr': (0.01, 0.3),
            'xgb_n_estimators': (50, 200),
            'xgb_max_depth': (3, 10),
        }

        optimizer = GeneticOptimizer(
            param_ranges=param_ranges,
            pop_size=10,
            generations=5,
            mutation_rate=0.1,
            tournament_size=3
        )

        best_params, best_loss = optimizer.optimize(X, y)
        print("\n最优参数:", best_params)
        print("最小验证损失:", best_loss)

        final_model = RobustDynamicBooster(
            base_input_dim=X.shape[1],
            n_estimators=best_params['booster_n_estimators'],
            base_lr=best_params['base_lr'],
            residual_steps=2,
            num_heads=4,
            xgb_max_depth=best_params['xgb_max_depth'],
            xgb_n_estimators=best_params['xgb_n_estimators'],
            xgb_reg_alpha=0.8,
            xgb_reg_lambda=1.2
        )
        final_model.fit(X, y)

        y_pred_train = final_model.predict(X)
        evaluate(y, y_pred_train, "训练集")

        if final_model.X_val is not None:
            y_pred_val = final_model.predict(final_model.X_val)
            evaluate(final_model.y_val, y_pred_val, "验证集")

    except Exception as e:
        print(f"错误发生: {str(e)}")
        print("常见问题排查：")
        print("1. 确认数据路径和列索引正确")
        print("2. 检查输入数据是否包含非数值或缺失值")
        print("3. 确保已安装所有依赖库")