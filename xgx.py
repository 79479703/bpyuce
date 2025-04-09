import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件（需安装openpyxl）
df = pd.read_excel('二胎.xlsx', engine='openpyxl').iloc[:, 1:16]  # 选取前11列

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 负号显示修正

# 创建带子图的画布
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# 计算三种相关系数矩阵
corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')
corr_kendall = df.corr(method='kendall')

# 绘制皮尔逊热力图
sns.heatmap(corr_pearson, annot=True, fmt=".2f",
            cmap='coolwarm', center=0,
            ax=axes[0], square=True,
            annot_kws={'size': 8}, cbar=True)
axes[0].set_title('Pearson 相关系数热力图', pad=20, fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# 绘制斯皮尔曼热力图
sns.heatmap(corr_spearman, annot=True, fmt=".2f",
            cmap='viridis', center=0,
            ax=axes[1], square=True,
            annot_kws={'size': 8}, cbar=True)
axes[1].set_title('Spearman 相关系数热力图', pad=20, fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# 绘制肯德尔热力图
heatmap = sns.heatmap(corr_kendall, annot=True, fmt=".2f",
                      cmap='YlOrRd', center=0,
                      ax=axes[2], square=True,
                      annot_kws={'size': 8}, cbar=True)
axes[2].set_title('Kendall 相关系数热力图', pad=20, fontsize=12)
axes[2].tick_params(axis='x', rotation=45)

# 调整布局并保存
plt.tight_layout(pad=3.0)
plt.savefig('correlation_heatmaps22.png', dpi=300, bbox_inches='tight')
plt.close()