import pandas as pd
from xgboost import XGBRegressor
from function import split_data

# 读取数据
data = pd.read_csv("../data/dataset.csv")
X_train, X_test, y_train, y_test = split_data(data)
X_virtual = pd.read_csv("synthetic_data.csv")

# 训练XGBoost回归模型
xgb = XGBRegressor(n_estimators=280,
                   learning_rate=0.09,
                   subsample=0.64,
                   gamma=0.3,
                   max_depth=18,
                   min_child_weight=9,
                   reg_alpha=0.5,
                   colsample_bytree=0.8,
                   colsample_bylevel=0.6,
                   colsample_bynode=0.6,
                   random_state=21)
xgb.fit(X_train, y_train)

y_train_pred = xgb.predict(X_train)
y_virtual_pred = xgb.predict(X_virtual)

# 将X_virtual和y_virtual_pred组合到一个DataFrame中
data_virtual = X_virtual.copy()
data_virtual['y_virtual_pred'] = y_virtual_pred

# 保存合成数据
data_virtual.to_csv('synthetic_data_with_predictions.csv', index=False)