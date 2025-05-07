import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import pickle
# 导入数据
res = pd.read_excel('新环境温度.xlsx').values

# 划分训练集和测试集
np.random.seed(0)  # 为了可重复性
temp = np.random.permutation(157)
P_train = res[temp[:120], 1:7].T
T_train = res[temp[:120], 7].T
M = P_train.shape[1]
P_test = res[temp[120:], 1:7].T
T_test = res[temp[120:], 7].T
N = P_test.shape[1]

# 单独限制每个输入变量的小数点位数
P_train[0, :] = np.round(P_train[0, :], 1)
P_train[1, :] = np.round(P_train[1, :], 1)
P_train[2, :] = np.round(P_train[2, :], 1)
P_train[3, :] = np.round(P_train[3, :], 1)
P_test[0, :] = np.round(P_test[0, :], 1)
P_test[1, :] = np.round(P_test[1, :], 1)
P_test[2, :] = np.round(P_test[2, :], 1)
P_test[3, :] = np.round(P_test[3, :], 1)

# 数据归一化
scaler_input = MinMaxScaler(feature_range=(0, 1))
p_train = scaler_input.fit_transform(P_train.T).T
scaler_output = MinMaxScaler(feature_range=(0, 1))
t_train = scaler_output.fit_transform(T_train.reshape(-1, 1)).flatten()
p_test = scaler_input.transform(P_test.T).T
t_test = scaler_output.transform(T_test.reshape(-1, 1)).flatten()

# 创建网络
net = MLPRegressor(hidden_layer_sizes=(10), max_iter=1000, tol=1e-6, learning_rate_init=0.08)

# 训练网络
net.fit(p_train.T, t_train)
t_sim1 = net.predict(p_train.T)
t_sim2 = net.predict(p_test.T)

# 数据反归一化
T_sim1 = scaler_output.inverse_transform(t_sim1.reshape(-1, 1)).flatten()
T_sim2 = scaler_output.inverse_transform(t_sim2.reshape(-1, 1)).flatten()

# 限制输出预测值的小数点位数
T_sim1_rounded = np.round(T_sim1, 2)
T_sim2_rounded = np.round(T_sim2, 2)

# 性能评估（训练集）
# 计算误差
error1 = T_train - T_sim1_rounded
# 计算 MAPE（平均绝对百分比误差）
mape1 = mean_absolute_percentage_error(T_train, T_sim1_rounded) * 100
# 计算 MAE（平均绝对误差）
mae1 = mean_absolute_error(T_train, T_sim1_rounded)
# 计算 MSE（均方误差）
mse1 = mean_squared_error(T_train, T_sim1_rounded)
# 计算 R²（决定系数）
r_squared1 = r2_score(T_train, T_sim1_rounded)

# 输出训练集评价指标
print(f'训练集 MAPE: {mape1}')
print(f'训练集 MAE: {mae1}')
print(f'训练集 MSE: {mse1}')
print(f'训练集 R²: {r_squared1}')

# 绘制训练集预测值与实际值的对比图
plt.figure(1)
plt.plot(np.arange(1, M + 1), T_train, '-*b', np.arange(1, M + 1), T_sim1_rounded, ':og')
plt.legend(['训练集实际值', '训练集预测值'], fontsize=10)
plt.title('训练集预测值与实际值对比', fontsize=12)
plt.xlabel('样本', fontsize=12)
plt.ylabel('值', fontsize=12)

# 性能评估（测试集）
# 计算误差
error2 = T_test - T_sim2_rounded
# 计算 MAPE（平均绝对百分比误差）
mape2 = mean_absolute_percentage_error(T_test, T_sim2_rounded) * 100
# 计算 MAE（平均绝对误差）
mae2 = mean_absolute_error(T_test, T_sim2_rounded)
# 计算 MSE（均方误差）
mse2 = mean_squared_error(T_test, T_sim2_rounded)
# 计算 R²（决定系数）
r_squared2 = r2_score(T_test, T_sim2_rounded)

# 输出测试集评价指标
print(f'测试集 MAPE: {mape2}')
print(f'测试集 MAE: {mae2}')
print(f'测试集 MSE: {mse2}')
print(f'测试集 R²: {r_squared2}')

# 绘制测试集预测值与实际值的对比图
plt.figure(2)
plt.plot(np.arange(1, N + 1), T_test, '-*b', np.arange(1, N + 1), T_sim2_rounded, ':og')
plt.legend(['测试集实际值', '测试集预测值'], fontsize=10)
plt.title('测试集预测结果对比', fontsize=12)
plt.xlabel('预测样本', fontsize=12)
plt.ylabel('产蛋量（个）', fontsize=12)

# 保存模型为 pkl 文件
with open('bpsjwl_model.pkl', 'wb') as file:
    pickle.dump(net, file)
print("模型已保存为bpsjwl_model.pkl")
# 修改后的模型保存代码（在原训练代码末尾添加）
# 保存归一化器
with open('scaler_input.pkl', 'wb') as f:
    pickle.dump(scaler_input, f)

with open('scaler_output.pkl', 'wb') as f:
    pickle.dump(scaler_output, f)


with open('scaler_info.pkl', 'wb') as f:
    pickle.dump({
        'input_min': scaler_input.data_min_,
        'input_max': scaler_input.data_max_,
        'output_min': scaler_output.data_min_,
        'output_max': scaler_output.data_max_
    }, f)

print("模型和归一化器已保存")