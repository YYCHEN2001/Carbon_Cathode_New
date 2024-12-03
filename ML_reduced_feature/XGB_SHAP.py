import pandas as pd
import shap
from matplotlib import pyplot as plt

from function import (split_data)

# 读取数据
data = pd.read_csv("../data/dataset_reduced.csv")
X_train, X_test, y_train, y_test = split_data(data, 'Cs')

from xgboost import XGBRegressor

xgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.17,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.5,
    'gamma': 0.05,
    'reg_alpha': 0.8,
    'reg_lambda': 5,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'random_state': 21
}
xgb = XGBRegressor(**xgb_params)
xgb.fit(X_train, y_train)
y_train_pred_xgb = xgb.predict(X_train)
y_test_pred_xgb = xgb.predict(X_test)

explainer = shap.Explainer(xgb)
shap_values = explainer(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar", plot_size=(17, 15), show=False)

# 获取特征重要性
importance = xgb.feature_importances_

# 使用 X_train 的列名称来代替特征名
feature_names = X_train.columns  # 直接使用原始数据的列名
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# 保存为 CSV 文件
importance_df.to_csv('output/xgb_feature_importance.csv', index=False)

# 获取当前图形对象
fig = plt.gcf()

# 放大字体大小并更改字体为Times New Roman且加粗
for ax in fig.axes:
    ax.title.set_fontsize(48)  # 放大标题字体
    ax.title.set_fontweight('bold')  # 加粗标题字体
    ax.title.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.xaxis.label.set_size(48)  # 放大 x 轴标签字体
    ax.xaxis.label.set_fontweight('bold')  # 加粗 x 轴标签字体
    ax.xaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.yaxis.label.set_size(48)  # 放大 y 轴标签字体
    ax.yaxis.label.set_fontweight('bold')  # 加粗 y 轴标签字体
    ax.yaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(48)  # 放大刻度标签字体
        label.set_fontweight('bold')  # 加粗刻度标签字体
        label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

plt.savefig('output/xgb_shap_train_summary_bar.png', bbox_inches='tight', pad_inches=0.1, transparent=False,
            dpi=600)

# 清除当前图形
plt.clf()

shap.summary_plot(shap_values, X_train, plot_size=(17, 15),
                  # max_display=10,
                  show=False)
# 获取当前图形对象
fig = plt.gcf()

# 放大字体大小
for ax in fig.axes:
    ax.title.set_fontsize(48)  # 放大标题字体
    ax.title.set_fontweight('bold')  # 加粗标题字体
    ax.title.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.xaxis.label.set_size(48)  # 放大 x 轴标签字体
    ax.xaxis.label.set_fontweight('bold')  # 加粗 x 轴标签字体
    ax.xaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.yaxis.label.set_size(48)  # 放大 y 轴标签字体
    ax.yaxis.label.set_fontweight('bold')  # 加粗 y 轴标签字体
    ax.yaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(48)  # 放大刻度标签字体
        label.set_fontweight('bold')  # 加粗刻度标签字体
        label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman
plt.savefig('output/xgb_shap_train_summary.png', bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=600)
