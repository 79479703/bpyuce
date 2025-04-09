import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

# 读取Excel文件
data = pd.read_excel('限制产奶量总上2.xlsx')

# 提取输入特征和输出标签（注意Python的0-based索引）
# iloc[:, 1:15] 表示第2列到第15列（包含15），共13列特征
# iloc[:, 16] 表示第17列作为目标
X = data.iloc[:, 1:13].values  # 输入特征
y = data.iloc[:, 13].values    # 输出标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建DMatrix数据集（优化内存使用和计算效率）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义基础XGBoost参数
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,            # 树深
    'learning_rate': 0.08,      # 学习率
    'n_estimators': 50       # 树的数量
}

# 训练模型
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=10
)

# 进行预测
y_pred = model.predict(dtest)
train_pred = model.predict(dtrain)
# 计算评估指标
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
r3 = r2_score(y_train,train_pred)
mae3 = mean_absolute_error(y_train, train_pred)
print(f'优化后的R²分数: {r2:.4f}')
print(f'优化后的MAE: {mae:.2f}')
print(f'优化后的R²分数: {r3:.4f}')
print(f'优化后的MAE: {mae3:.2f}')


# 特征重要性分析（辅助理解模型）
importance = model.get_score(importance_type='weight')
print("\n特征重要性：")
for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"特征{k}: {v}")
