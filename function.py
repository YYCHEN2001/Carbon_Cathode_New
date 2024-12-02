import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

def calculate_metrics(y_true, y_pred):
    """
    Calculate and return actual vs pred fig for data_dopants metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


def metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, model_name):
    R2_train, MAE_train, MAPE_train, RMSE_train = calculate_metrics(y_train, y_train_pred)
    R2_test, MAE_test, MAPE_test, RMSE_test = calculate_metrics(y_test, y_test_pred)
    metrics = {'model': model_name,
                   'R2_train': R2_train, 'MAE_train': MAE_train, 'MAPE_train': MAPE_train, 'RMSE_train': RMSE_train,
                   'R2_test': R2_test, 'MAE_test': MAE_test, 'MAPE_test': MAPE_test, 'RMSE_test': RMSE_test}
    model_name_df = pd.DataFrame([metrics])
    return model_name_df

def plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, figtitle, figpath=None):
    """
    Plot the actual vs predicted values for both training and test sets,
    and plot y=x as the fit line.
    """
    # 设置全局字体为Times New Roman，字号为32，字体粗细为粗体
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 32,
        'font.weight': 'bold',
        'figure.figsize': (12, 12)
    })

    # 绘制训练集和测试集的散点图
    plt.scatter(y_train, y_pred_train, color='blue', label='Train', s=50, alpha=0.5)
    plt.scatter(y_test, y_pred_test, color='red', label='Test', s=50, alpha=0.5)

    # 计算合并数据的最小值和最大值，用于设置坐标轴范围和绘制y=x线
    y_pred_train = y_pred_train.ravel()
    y_pred_test = y_pred_test.ravel()
    y_combined = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])
    min_val, max_val = np.min(y_combined), np.max(y_combined)
    padding = (max_val - min_val) * 0.05
    padded_min, padded_max = min_val - padding, max_val + padding

    # 绘制y=x的虚线，线宽为3
    plt.plot([padded_min, padded_max], [padded_min, padded_max], 'k--', lw=3, label='Regression Line')

    # 设置标题和轴标签，明确指定加粗
    plt.title(figtitle, fontweight='bold', pad=20) # pad=20避免标题和轴线重合
    plt.xlabel('Actual Values', fontweight='bold')
    plt.ylabel('Predicted Values', fontweight='bold')

    # 设置图例，无边框，位于左上角
    plt.legend(frameon=False, loc='upper left', fontsize=28)

    # 设置坐标轴为相同比例，并且坐标轴范围一致
    plt.axis('equal')
    plt.xlim([padded_min, padded_max])
    plt.ylim([padded_min, padded_max])

    # 设置刻度线的长度和粗细
    plt.tick_params(axis='both', which='major', length=10, width=2, labelsize=32)

    # 检查并统一X轴和Y轴的刻度
    # 可以通过设置两个轴的相同刻度，或者根据数据自动选择刻度
    x_ticks = np.arange(0, max(y_combined) + 1, 50)  # 可以根据数据范围调整
    y_ticks = np.arange(0, max(y_combined) + 1, 50)  # 使得X和Y轴的刻度间隔相同

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # 设置图形边界的宽度和可见性
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('black')

    # 保存图像，背景透明，紧凑布局
    plt.savefig(figpath, bbox_inches='tight', transparent=True)
    plt.show()


def split_data(data, target):
    data['target_class'] = pd.qcut(data[target], q=10, labels=False)
    X = data.drop([target, 'target_class'], axis=1)
    y = data[target]
    stratify_column = data['target_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)
    return X_train, X_test, y_train, y_test

def save_pdp(model, X_train, feature_name, savepath):
    # 计算部分依赖数据
    pdp_result = PartialDependenceDisplay.from_estimator(model, X_train, [feature_name], grid_resolution=100)
    # 提取部分依赖数据
    feature_values = pdp_result.lines_[0][0].get_xdata()  # X轴特征值
    partial_dependence_values = pdp_result.lines_[0][0].get_ydata()  # Y轴部分依赖值
    # 创建DataFrame并保存为CSV
    pdp_data = pd.DataFrame({
        'Feature Values': feature_values,
        'Partial Dependence Values': partial_dependence_values
    })
    # 保存CSV到本地
    pdp_data.to_csv(savepath, index=False)

def save_2d_pdp(model, X_train, features, savepath):
    # %%
    # 计算部分依赖数据
    # 检查特征是否在数据集中
    print("Training data columns:", X_train.columns)
    assert all([feature in X_train.columns for feature in features]), "One or more features not found in training data"

    # 计算2D部分依赖数据
    try:
        fig, ax = plt.subplots(figsize=(12, 9))
        pdp_display = PartialDependenceDisplay.from_estimator(model, X_train, [features], ax=ax, grid_resolution=50)

        # 检查是否成功生成部分依赖数据
        if len(pdp_display.pd_results) == 0:
            raise ValueError("No partial dependence data was generated for the given features.")

        # 提取网格值和部分依赖值
        grid_values = pdp_display.pd_results[0].grid_values
        x_values, y_values = grid_values
        average_values = pdp_display.pd_results[0].average

        # 创建网格值
        X_mesh, Y_mesh = np.meshgrid(x_values, y_values)

        # 将数据存储为DataFrame
        pdp_2d_data = pd.DataFrame({
            f'{features[0]}': X_mesh.flatten(),
            f'{features[1]}': Y_mesh.flatten(),
            'Partial Dependence': average_values.transpose().flatten()
        })

        # 保存为CSV文件
        pdp_2d_data.to_csv(savepath, index=False)

        # 显示保存路径
        print("2D PDP 数据已保存到本地")

        # 显示图形
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print(f"Error in generating partial dependence plot: {e}")

    except IndexError as e:
        print(f"IndexError: {e}. Please check if the features are correctly specified.")