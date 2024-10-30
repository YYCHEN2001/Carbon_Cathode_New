import pandas as pd

from function import metrics_to_dataframe, plot_actual_vs_predicted, split_data

# 读取数据
data = pd.read_csv("../data/dataset.csv")
X_train, X_test, y_train, y_test = split_data(data)

from sklearn.linear_model import LinearRegression

# 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

lr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Linear Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Linear Regression', 'lr.png')

from sklearn.svm import SVR

# 训练支持向量回归模型
svr = SVR(
    C=6,
    kernel='poly',
    degree=7,
    gamma='scale',
    coef0=5,
    epsilon=0.75,
    verbose=True
)

svr.fit(X_train, y_train)

y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

svr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Support Vector Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Support Vector Regression',
                         'svr.png')

from sklearn.ensemble import RandomForestRegressor

# 训练随机森林回归模型
rfr = RandomForestRegressor(n_estimators=50,
                            max_depth=32,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            max_features=1,
                            random_state=21)
rfr.fit(X_train, y_train)

y_train_pred = rfr.predict(X_train)
y_test_pred = rfr.predict(X_test)

rfr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Randon Forest Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Random Forest Regression',
                         'rfr.png')

from sklearn.ensemble import GradientBoostingRegressor

# 训练梯度提升回归模型
gbr = GradientBoostingRegressor(n_estimators=900,
                                alpha=0.001,
                                learning_rate=0.04,
                                max_depth=46,
                                max_features=0.39,
                                min_samples_leaf=5,
                                min_samples_split=12,
                                subsample=0.33,
                                random_state=21)
gbr.fit(X_train, y_train)

y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)

gbr_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'Gradient Boosting Regression')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'Gradient Boosting Regression',
                         'gbr.png')

from xgboost import XGBRegressor

# 训练XGBoost回归模型
xgb = XGBRegressor(n_estimators=800,
                   learning_rate=0.04,
                   subsample=0.49,
                   gamma=0.6,
                   max_depth=14,
                   min_child_weight=6,
                   reg_alpha=0.5,
                   colsample_bytree=0.8,
                   colsample_bylevel=0.8,
                   colsample_bynode=0.6,
                   random_state=21)
xgb.fit(X_train, y_train)

y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)

xgb_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'XGBoost')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'XGBoost', 'XGB.png')

from lightgbm import LGBMRegressor

# 训练LightGBM回归模型
lgbm = LGBMRegressor(
    colsample_bytree=0.25,
    learning_rate=0.05,
    max_depth=17,
    min_child_samples=11,
    n_estimators=900,
    num_leaves=139,
    reg_alpha=0.36,
    reg_lambda=0.2,
    subsample=0.6,
    random_state=21
)

lgbm.fit(X_train, y_train)

y_train_pred = lgbm.predict(X_train)
y_test_pred = lgbm.predict(X_test)

lgbm_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'LightGBM')
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, 'LightGBM', 'LightGBM.png')

# 将所有模型的评估指标合并为一个DataFrame
metrics = pd.concat([lr_metrics, svr_metrics, rfr_metrics, gbr_metrics, xgb_metrics, lgbm_metrics])
metrics_rounded = metrics.round(3)
metrics_rounded.to_markdown('report_models.md', index=False, tablefmt='github')
print(metrics_rounded)
