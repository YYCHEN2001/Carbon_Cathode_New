import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from function import split_data, metrics_to_dataframe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv("dataset_reduced.csv")
X_train, X_test, y_train, y_test = split_data(data, 'Cs')

# 数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化 TabPFN 回归器
model = TabPFNRegressor(device=device)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算并显示评估指标
tabpfn_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'tabPFN')

print(tabpfn_metrics)
