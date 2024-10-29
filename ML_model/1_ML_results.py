import pandas as pd

from function import (split_data)

# 读取数据
data = pd.read_csv("../data/dataset.csv")
X_train_scaled, X_test_scaled, y_train, y_test = split_data(data)

from function import metrics_to_dataframe, plot_actual_vs_predicted
from sklearn.linear_model import LinearRegression

# 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)

lr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Linear Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Linear Regression', 'lr.png')

from sklearn.svm import SVR

# 训练支持向量回归模型
svr = SVR(
    C=5,
    kernel='poly',
    degree=5,
    gamma='scale',
    coef0=5,
    epsilon=0.75,
    verbose=True
)

svr.fit(X_train_scaled, y_train)

y_train_pred = svr.predict(X_train_scaled)
y_test_pred = svr.predict(X_test_scaled)

svr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Support Vector Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Support Vector Regression',
                         'svr.png')

from sklearn.ensemble import RandomForestRegressor

# 训练随机森林回归模型
rfr = RandomForestRegressor(n_estimators=100,
                            max_depth=10,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            max_features=1,
                            random_state=21)
rfr.fit(X_train_scaled, y_train)

y_train_pred = rfr.predict(X_train_scaled)
y_test_pred = rfr.predict(X_test_scaled)

rfr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Randon Forest Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Random Forest Regression',
                         'rfr.png')

from sklearn.ensemble import GradientBoostingRegressor

# 训练梯度提升回归模型
gbr = GradientBoostingRegressor(n_estimators=270,
                                alpha=0.001,
                                learning_rate=0.1,
                                max_depth=17,
                                max_features=0.89,
                                min_samples_leaf=9,
                                min_samples_split=4,
                                subsample=0.68,
                                random_state=21)
gbr.fit(X_train_scaled, y_train)

y_train_pred = gbr.predict(X_train_scaled)
y_test_pred = gbr.predict(X_test_scaled)

gbr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Gradient Boosting Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Gradient Boosting Regression',
                         'gbr.png')

from xgboost import XGBRegressor

# 训练XGBoost回归模型
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

y_train_pred = xgb.predict(X_train_scaled)
y_test_pred = xgb.predict(X_test_scaled)

xgb_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'XGBoost')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'XGBoost', 'XGB.png')

from lightgbm import LGBMRegressor

# 训练LightGBM回归模型
lgbm = LGBMRegressor(
    colsample_bytree=0.5,
    learning_rate=0.12,
    max_depth=25,
    min_child_samples=12,
    n_estimators=240,
    num_leaves=95,
    reg_alpha=0.13,
    reg_lambda=0.23,
    subsample=0.95,
    random_state=21
)

lgbm.fit(X_train_scaled, y_train)

y_train_pred = lgbm.predict(X_train_scaled)
y_test_pred = lgbm.predict(X_test_scaled)

lgbm_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'LightGBM')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'LightGBM', 'LightGBM.png')

# 将所有模型的评估指标合并为一个DataFrame
metrics = pd.concat([lr_metrics, svr_metrics, rfr_metrics, gbr_metrics, lgbm_metrics, xgb_metrics])
metrics_rounded = metrics.round(3)
metrics_rounded.to_markdown('report_models.md', index=False, tablefmt='github')
print(metrics_rounded)
