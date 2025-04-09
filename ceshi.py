import streamlit as st
import pandas as pd
import pickle
import os
import shap
import numpy as np


class BPNN:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        # 输入特征调整为7个
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def predict(self, data):
        # 确保输入数据为二维数组
        if isinstance(data, pd.DataFrame):
            input_array = data.values.astype('float32')
        elif isinstance(data, np.ndarray):
            input_array = data.astype('float32')
        else:
            raise ValueError("输入数据必须是 DataFrame 或 NumPy 数组")

        # 校验输入维度为 (样本数, 7)
        if input_array.shape[1] != 7:
            raise ValueError(f"输入特征应为7个，当前为 {input_array.shape[1]} 个")

        return self.model.predict(input_array)


# 加载模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'bpsjwl_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 页面布局
st.title("基于BP神经网络的产蛋量预测")
st.sidebar.header("环境参数输入")

# 定义7个输入特征
inputs = {
    "最高温度": st.sidebar.slider("最高温度", min_value=10, max_value=30, value=20),
    "最低温度": st.sidebar.slider("最低温度", min_value=0, max_value=10, value=5),
    "平均温度": st.sidebar.slider("平均温度", min_value=5, max_value=25, value=15),
    "湿度": st.sidebar.slider("湿度", min_value=20, max_value=100, value=60),
    "光照强度": st.sidebar.slider("光照强度", min_value=10, max_value=100, value=50),
    "二氧化碳浓度": st.sidebar.slider("二氧化碳浓度", min_value=200, max_value=500, value=350),
    "氨气浓度": st.sidebar.slider("氨气浓度", min_value=0, max_value=10, value=3)
}

# 转换为DataFrame
input_data = pd.DataFrame([inputs])

if st.button("预测产蛋量"):
    try:
        prediction = model.predict(input_data)
        st.success(f"预测产蛋量: {prediction[0]:.2f} 枚")

        # 使用KernelExplainer替代DeepExplainer
        explainer = shap.KernelExplainer(model.predict, input_data)
        shap_values = explainer.shap_values(input_data)

        # 可视化代码保持不变
    except Exception as e:
        st.error(f"预测失败: {str(e)}")

        # 可视化
        st.subheader("特征影响力图")
        shap_html = shap.force_plot(
            explainer.expected_value[0],
            shap_values[0][0],
            input_data.iloc[0],
            feature_names=input_data.columns
        )
        st.components.v1.html(shap_html._repr_html_(), height=500)

    except Exception as e:
        st.error(f"预测失败: {str(e)}")