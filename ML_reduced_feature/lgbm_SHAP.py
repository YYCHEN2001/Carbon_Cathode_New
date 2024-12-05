import pandas as pd
import shap
from matplotlib import pyplot as plt
from lgbm_function import trained_lgbm
from function import (split_data)

# 读取数据
data = pd.read_csv("../data/dataset_reduced.csv")
X_train, X_test, y_train, y_test = split_data(data, 'Cs')
lgbm = trained_lgbm(X_train, y_train)
y_train_pred_lgbm = lgbm.predict(X_train)
y_test_pred_lgbm = lgbm.predict(X_test)

explainer = shap.Explainer(lgbm)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", plot_size=(17, 15),
                  show=False)
# 获取当前图形对象
fig = plt.gcf()

# 放大字体大小并更改字体为Times New Roman且加粗
for ax in fig.axes:
    ax.title.set_fontsize(48)  # 放大标题字体
    ax.title.set_fontweight('bold')  # 加粗标题字体
    ax.title.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.xaxis.label.set_size(32)  # 放大 x 轴标签字体
    ax.xaxis.label.set_fontweight('bold')  # 加粗 x 轴标签字体
    ax.xaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.yaxis.label.set_size(36)  # 放大 y 轴标签字体
    ax.yaxis.label.set_fontweight('bold')  # 加粗 y 轴标签字体
    ax.yaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(32)  # 放大刻度标签字体
        label.set_fontweight('bold')  # 加粗刻度标签字体
        label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

plt.savefig('output/lgbm_shap_train_summary_bar.png', bbox_inches='tight', pad_inches=0.1, transparent=False,
            dpi=600)

# 清除当前图形
plt.clf()

shap.summary_plot(shap_values, X_train, plot_size=(17, 15),
                  show=False)
# 获取当前图形对象
fig = plt.gcf()

# 放大字体大小
for ax in fig.axes:
    ax.title.set_fontsize(48)  # 放大标题字体
    ax.title.set_fontweight('bold')  # 加粗标题字体
    ax.title.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.xaxis.label.set_size(32)  # 放大 x 轴标签字体
    ax.xaxis.label.set_fontweight('bold')  # 加粗 x 轴标签字体
    ax.xaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    ax.yaxis.label.set_size(36)  # 放大 y 轴标签字体
    ax.yaxis.label.set_fontweight('bold')  # 加粗 y 轴标签字体
    ax.yaxis.label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(32)  # 放大刻度标签字体
        label.set_fontweight('bold')  # 加粗刻度标签字体
        label.set_fontname('Times New Roman')  # 设置字体为 Times New Roman
plt.savefig('output/lgbm_shap_train_summary.png', bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=600)

# Get feature importance
feature_importance = lgbm.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})
# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Save to CSV
importance_df.to_csv('output/feature_importance.csv', index=False)