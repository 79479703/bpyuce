import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import shap
from io import BytesIO

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
mode = st.sidebar.selectbox("选择预测模式", ["单样本预测", "批量预测"])

# 定义特征顺序
feature_order = ["最高温度", "最低温度", "湿度", "光照强度", "二氧化碳浓度", "氨气浓度"]

if mode == "单样本预测":
    # 单样本预测界面
    st.sidebar.header("环境参数输入")

    inputs = {
        "最高温度": round(st.sidebar.slider("最高温度", 13, 19, 16) + np.random.rand() * 0.1, 1),
        "最低温度": round(st.sidebar.slider("最低温度", 8, 15, 12) + np.random.rand() * 0.1, 1),
        "湿度": round(st.sidebar.slider("湿度", 49, 79, 60) + np.random.rand() * 0.1, 1),
        "光照强度": round(st.sidebar.slider("光照强度", 5, 15, 10) + np.random.rand() * 0.1, 1),
        "二氧化碳浓度": st.sidebar.slider("二氧化碳浓度", 450, 625, 550),
        "氨气浓度": st.sidebar.slider("氨气浓度", 0, 5, 3)
    }

    input_df = pd.DataFrame([inputs])[feature_order]
    input_df.iloc[:, :4] = input_df.iloc[:, :4].apply(lambda x: np.round(x, 1))

    # 数据范围检查
    for i in range(6):
        if input_df.iloc[0, i] < input_min[i]:
            st.warning(f"{feature_order[i]}低于训练数据最小值({input_min[i]}), 预测可能不准确")
        if input_df.iloc[0, i] > input_max[i]:
            st.warning(f"{feature_order[i]}超过训练数据最大值({input_max[i]}), 预测可能不准确")

    if st.button("预测产蛋量"):
        try:
            scaled_input = scaler_input.transform(input_df)
            scaled_pred = model.predict(scaled_input.reshape(1, -1))  # 关键修改点
            final_pred = scaler_output.inverse_transform(scaled_pred.reshape(-1, 1))

            if final_pred[0][0] < 0:
                final_pred[0][0] = 0
                st.warning("预测值出现负数，已自动校正为0")

            prediction = int(round(final_pred[0][0]))
            st.success(f"预测产蛋量: {prediction} 枚")

        except Exception as e:
            st.error(f"预测失败: {str(e)}")

else:
    # 批量预测界面
    st.header("批量预测（Excel文件）")
    uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx"])
    github_url = st.text_input("或输入GitHub文件URL（原始文件链接）")

    if uploaded_file or github_url:
        try:
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_excel(github_url)

            if list(df.columns) != feature_order:
                st.error(f"文件列名必须为：{feature_order}")
                st.stop()

            processed_df = df.copy()
            processed_df.iloc[:, :4] = processed_df.iloc[:, :4].round(1)

            # 数据范围检查
            out_of_range = {}
            for i, col in enumerate(feature_order):
                min_val = input_min[i]
                max_val = input_max[i]

                below = processed_df[col] < min_val
                above = processed_df[col] > max_val

                if below.any() or above.any():
                    out_of_range[col] = {
                        'min': min_val,
                        'max': max_val,
                        'below_rows': processed_df.index[below].tolist(),
                        'above_rows': processed_df.index[above].tolist()
                    }

            if out_of_range:
                warning_msg = "以下数据超出训练范围:\n"
                for col, info in out_of_range.items():
                    warning_msg += f"- {col}（范围：{info['min']}~{info['max']}）"
                    parts = []
                    if info['below_rows']:
                        parts.append(f"行{info['below_rows']}低于最小值")
                    if info['above_rows']:
                        parts.append(f"行{info['above_rows']}高于最大值")
                    warning_msg += "：" + "，".join(parts) + "\n"
                st.warning(warning_msg)

            # 关键修改点：确保输入维度正确
            scaled_input = scaler_input.transform(processed_df)
            if len(scaled_input.shape) == 1:  # 处理单样本情况
                scaled_input = scaled_input.reshape(1, -1)
            scaled_pred = model.predict(scaled_input)
            final_pred = scaler_output.inverse_transform(scaled_pred.reshape(-1, 1))

            final_pred[final_pred < 0] = 0
            predictions = np.round(final_pred).astype(int)

            result_df = df.copy()
            result_df["预测产蛋量"] = predictions

            st.subheader("预测结果")
            st.dataframe(result_df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                label="下载预测结果",
                data=output.getvalue(),
                file_name="预测结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"处理文件时出错：{str(e)}")