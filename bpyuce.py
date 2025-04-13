# 修改后的streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import shap

# 加载模型和归一化器
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_object(filename):
    with open(os.path.join(current_dir, filename), 'rb') as file:
        return pickle.load(file)


model = load_object('bpsjwl_model.pkl')
scaler_input = load_object('scaler_input.pkl')
scaler_output = load_object('scaler_output.pkl')

# 获取训练数据范围用于验证
input_min = scaler_input.data_min_
input_max = scaler_input.data_max_

# 页面布局
st.title("基于BP神经网络的产蛋量预测")
st.sidebar.header("环境参数输入")

# 定义输入特征（确保顺序与训练数据完全一致）
feature_order = ["最高温度", "最低温度", "湿度", "光照强度", "二氧化碳浓度", "氨气浓度"]

# 获取用户输入（添加小数处理）
inputs = {
    "最高温度": round(st.sidebar.slider("最高温度", 13, 19, 16) + np.random.rand() * 0.1, 1),
    "最低温度": round(st.sidebar.slider("最低温度", 8, 15, 12) + np.random.rand() * 0.1, 1),
    "湿度": round(st.sidebar.slider("湿度", 49, 79, 60) + np.random.rand() * 0.1, 1),
    "光照强度": round(st.sidebar.slider("光照强度", 5, 15, 10) + np.random.rand() * 0.1, 1),
    "二氧化碳浓度": st.sidebar.slider("二氧化碳浓度", 450, 625, 550),
    "氨气浓度": st.sidebar.slider("氨气浓度", 0, 5, 3)
}

# 转换为DataFrame并验证顺序
input_df = pd.DataFrame([inputs])[feature_order]

# 数据预处理（与训练时一致）
input_df.iloc[:, :4] = input_df.iloc[:, :4].apply(lambda x: np.round(x, 1))  # 前四列保留1位小数

# 数据范围检查
for i in range(6):
    if input_df.iloc[0, i] < input_min[i]:
        st.warning(f"{feature_order[i]}低于训练数据最小值({input_min[i]}), 预测可能不准确")
    if input_df.iloc[0, i] > input_max[i]:
        st.warning(f"{feature_order[i]}超过训练数据最大值({input_max[i]}), 预测可能不准确")

if st.button("预测产蛋量"):
    try:
        # 数据归一化
        scaled_input = scaler_input.transform(input_df)

        # 进行预测
        scaled_pred = model.predict(scaled_input)

        # 反归一化处理
        final_pred = scaler_output.inverse_transform(scaled_pred.reshape(-1, 1))

        # 合理性校验
        if final_pred[0][0] < 0:
            st.warning("预测值出现负数，已自动校正为0")
            final_pred[0][0] = 0

        prediction = int(round(final_pred[0][0]))

        st.success(f"预测产蛋量: {prediction} 枚")


    except Exception as e:
        st.error(f"预测失败: {str(e)}")