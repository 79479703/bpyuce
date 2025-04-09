import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 1. 数据预处理
def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, 1:8].values  # 2-8列为输入特征
    y = df.iloc[:, 8].values  # 9列为输出
    return X, y


# 2. 数据标准化
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# 3. 贝叶斯优化目标函数
def bp_objective(learning_rate, hidden_units, epochs, l2_reg):
    model = Sequential([
        Dense(int(hidden_units), activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
              input_shape=(7,)),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])

    # 使用部分验证集进行快速评估
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42)

    history = model.fit(X_train_sub, y_train_sub,
                        epochs=int(epochs),
                        batch_size=32,
                        verbose=0,
                        validation_data=(X_val, y_val))

    # 返回验证集的RMSE（贝叶斯优化将最大化该负值）
    return -np.sqrt(history.history['val_loss'][-1])


# 4. 主程序
if __name__ == "__main__":
    # 加载数据
    X, y = load_data("新环境温度.xlsx")  # 替换为实际路径
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # 定义超参数搜索空间[4,6](@ref)
    pbounds = {
        'learning_rate': (0.0001, 0.1),
        'hidden_units': (16, 128),
        'epochs': (50, 300),
        'l2_reg': (0.0001, 0.1)
    }

    # 初始化贝叶斯优化器[4](@ref)
    optimizer = BayesianOptimization(
        f=bp_objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # 执行优化[4,6](@ref)
    optimizer.maximize(init_points=5, n_iter=20)

    # 获取最优参数
    best_params = optimizer.max['params']
    print("\n最优参数组合：", best_params)

    # 用最优参数训练最终模型
    final_model = Sequential([
        Dense(int(best_params['hidden_units']), activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(best_params['l2_reg']),
              input_shape=(7,)),
        Dense(1)
    ])

    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                        loss='mse')

    history = final_model.fit(X_train_scaled, y_train,
                              epochs=int(best_params['epochs']),
                              batch_size=32,
                              verbose=0)

    # 模型评估[7](@ref)
    y_pred = final_model.predict(X_test_scaled).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n测试集评估结果：\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}")

    # 绘制预测对比图[7,8](@ref)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='真实值', marker='o', linestyle='--')
    plt.plot(y_pred, label='预测值', marker='x', alpha=0.7)
    plt.title("预测值与真实值对比")
    plt.xlabel("样本序号")
    plt.ylabel("目标值")
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison.png', dpi=300)
    plt.show()