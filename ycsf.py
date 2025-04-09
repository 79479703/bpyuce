import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# 读取Excel文件
data = pd.read_excel('相关性.xlsx')

# 提取输入特征和输出标签（注意Python的0-based索引）
# iloc[:, 1:15] 表示第2列到第14列（包含14），共13列特征
# iloc[:, 16] 表示第17列作为目标
X = data.iloc[:, 1:16].values  # 输入特征
y = data.iloc[:, 16].values  # 输出标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建DMatrix数据集（优化内存使用和计算效率）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义超参数搜索空间
space = [
    Integer(50, 200, name='n_estimators'),  # 树的数量
    Real(0.001, 0.1, name='learning_rate'),  # 学习率
    Integer(3, 10, name='max_depth'),  # 树深
    Integer(50, 200, name='num_boost_round')  # 学习轮数
]


@use_named_args(space)
def objective(**params):
    model = xgb.train(
        {
            'objective': 'reg:squarederror',
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'n_estimators': params['n_estimators']
        },
        dtrain,
        num_boost_round=params['num_boost_round'],
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False
    )
    y_pred = model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    # 我们的目标是最小化 MAE，因此直接返回 MAE
    return mae


# 使用遗传算法进行超参数优化
result = gp_minimize(objective, space, n_calls=50, random_state=42)

# 输出最优超参数
best_n_estimators = result.x[0]
best_learning_rate = result.x[1]
best_max_depth = result.x[2]
best_num_boost_round = result.x[3]
print(f"最优树的数量: {best_n_estimators}")
print(f"最优学习率: {best_learning_rate}")
print(f"最优树深: {best_max_depth}")
print(f"最优学习轮数: {best_num_boost_round}")

# 使用最优超参数重新训练模型
best_params = {
    'objective': 'reg:squarederror',
    'max_depth': best_max_depth,
    'learning_rate': best_learning_rate,
    'n_estimators': best_n_estimators
}
best_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_num_boost_round,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=10
)

# 进行预测
y_pred = best_model.predict(dtest)
train_pred = best_model.predict(dtrain)

# 计算评估指标
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r3 = r2_score(y_train, train_pred)
mae3 = mean_absolute_error(y_train, train_pred)

print(f'优化后的R²分数: {r2:.4f}')
print(f'优化后的MAE: {mae:.2f}')
print(f'训练集R²分数: {r3:.4f}')
print(f'训练集MAE: {mae3:.2f}')

# 特征重要性分析（辅助理解模型）
importance = best_model.get_score(importance_type='weight')
print("\n特征重要性：")
for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"特征{k}: {v}")
