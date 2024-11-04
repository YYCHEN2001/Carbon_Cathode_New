import pandas as pd
from xgboost import XGBRegressor

from function import split_data

# 读取数据
data = pd.read_csv("../data/dataset.csv")
X_train, X_test, y_train, y_test = split_data(data)
data_real = pd.read_excel("../data/data_real.xlsx")
X_real = data_real.drop(columns=['Name', 'Cs'], axis=1)
y_real = data_real['Cs']

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
y_test_pred = xgb.predict(X_test)
y_real_pred = xgb.predict(X_real)

# 将真实数据的X，y，y_real_pred预测结果合并到一个表格保存到csv文件
data_real['Cs_pred'] = y_real_pred
data_real.to_csv("../data/data_real_pred.csv", index=False)
