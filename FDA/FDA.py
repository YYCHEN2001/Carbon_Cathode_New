import pandas as pd

# 读取数据
data = pd.read_csv("../data/dataset.csv")
import seaborn as sns
import matplotlib.pyplot as plt

# 为了避免错误输出，先将'ID/IG'列更名为'ID_IG'
data_violin = data.rename(columns={'ID/IG': 'IDperIG'})

# 设置图像风格
sns.set_theme(style="whitegrid")

# 设置字体
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
    'font.weight': 'bold'
})

# 为每一列绘制和保存提琴图
for col in data_violin.columns:
    # 设置图像尺寸
    plt.figure(figsize=(6, 10))

    # 绘制提琴图
    sns.violinplot(data=data_violin[col])

    # 设置标题和标签
    plt.title(f"Distribution of {col}", weight='bold', fontsize=24)
    plt.xlabel(col, weight='bold', fontsize=24)
    plt.ylabel("Values", weight='bold', fontsize=24)

    # 保存图像
    plt.savefig(f'violin_{col}.png', dpi=300, bbox_inches='tight')

    # 显示图像（可选）
    # plt.show()

    # 关闭当前图像，释放内存
    plt.close()

import numpy as np
# 计算相关系数
corr = data.corr()

# 生成下三角矩阵，只保留下三角部分，上三角部分设为 NaN
mask = np.triu(np.ones_like(corr, dtype=bool))
corr[mask] = np.nan

# 手动设置 Seaborn 主题，并控制字体大小
sns.set_theme(style='white', rc={
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'font.size': 24
})

# 设置图像尺寸
plt.figure(figsize=(24, 20))

# 生成热力图，不显示背景，手动调整字体大小
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .8}, square=True,
    annot_kws={"size": 24},  # 设置数字的字体大小
)

# 手动设置 x 和 y 轴标签的字体大小
plt.xticks(fontsize=24, weight='bold', fontname='Times New Roman')
plt.yticks(fontsize=24, weight='bold', fontname='Times New Roman')

# 保存图像
plt.savefig('pearson_corr.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 数据分布统计
statistics = data.describe(include='all')
statistics_transposed = statistics.transpose().round(2)

# 将统计结果保存到 Markdown 文件
with open('statistics.md', 'w') as f:
    f.write(statistics_transposed.to_markdown())
