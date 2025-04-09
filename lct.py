import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow

fig, ax = plt.subplots(figsize=(10, 12))

# 绘制模块
modules = {
    "Input": (5, 10),
    "EnhancedFeatureProcessor": (5, 8),
    "MultiHeadAttention": (3, 6),
    "Residual+LayerNorm": (5, 6),
    "FC+GELU+Sigmoid": (7, 6),
    "DynamicFeature": (5, 4),
    "HybridXGBModel": (5, 2),
    "RobustDynamicBooster": (8, 2)
}

# 绘制模块框
for name, (x, y) in modules.items():
    ax.add_patch(Rectangle((x-2, y-0.5), 4, 1, edgecolor='black', facecolor='lightgray'))
    plt.text(x, y, name, ha='center', va='center')

# 绘制连接线
connections = [
    ("Input", "EnhancedFeatureProcessor"),
    ("EnhancedFeatureProcessor", "MultiHeadAttention"),
    ("EnhancedFeatureProcessor", "Residual+LayerNorm"),
    ("EnhancedFeatureProcessor", "FC+GELU+Sigmoid"),
    ("MultiHeadAttention", "Residual+LayerNorm"),
    ("Residual+LayerNorm", "FC+GELU+Sigmoid"),
    ("FC+GELU+Sigmoid", "DynamicFeature"),
    ("DynamicFeature", "HybridXGBModel"),
    ("HybridXGBModel", "RobustDynamicBooster")
]

for start, end in connections:
    sx, sy = modules[start]
    ex, ey = modules[end]
    ax.annotate("", xy=(ex, ey+0.5), xytext=(sx, sy-0.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))

plt.axis('off')
plt.show()
