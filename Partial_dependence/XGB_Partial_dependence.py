import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 读取数据
data = pd.read_csv("../data/dataset.csv")
data['target_class'] = pd.qcut(data['Cs'], q=10, labels=False)
X = data.drop(['Cs', 'target_class'], axis=1)
y = data['Cs']
stratify_column = data['target_class']

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

# 标准化特征数据
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 DataFrame 以保持特征名称
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# 训练 XGBoost 回归模型
xgb = XGBRegressor(n_estimators=240,
                   learning_rate=0.1,
                   subsample=0.66,
                   gamma=1,
                   max_depth=22,
                   min_child_weight=3,
                   reg_alpha=1,
                   colsample_bytree=0.6,
                   colsample_bylevel=0.6,
                   colsample_bynode=0.6,
                   random_state=21)
xgb.fit(X_train_scaled, y_train)

# 选择要绘制部分依赖图的特征
features = list(X.columns)

# 绘制每个特征的部分依赖图
for feature in features:
    try:
        # 使用 partial_dependence 计算部分依赖
        pd_result = partial_dependence(xgb, X=X_train_scaled, features=[feature], grid_resolution=100)

        # 获取标准化特征值和部分依赖结果
        x_vals_scaled = pd_result["values"][0]  # 网格点的特征值
        y_vals = pd_result["average"][0]  # 对应的部分依赖结果

        # 将横坐标值还原为原始尺度
        x_vals_original = scaler.inverse_transform(
            np.array([x_vals_scaled if f == feature else np.zeros_like(x_vals_scaled) for f in X.columns]).T
        )[:, X.columns.get_loc(feature)]

        # 绘制部分依赖图
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals_original, y_vals, label=f'Partial Dependence of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial Dependence')
        ax.set_title(f'Partial Dependence of {feature}')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        # 保存图像到当前目录下
        filename = f'xgb_partial_dependence_{feature}.png'
        plt.savefig(filename, format='png', dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"Could not plot partial dependence for feature {feature}: {e}")
