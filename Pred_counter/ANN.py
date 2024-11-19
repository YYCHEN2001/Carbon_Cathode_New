import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numba.core.typing.builtins import Print
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 自定义 MAPE 损失函数
class MAPE_Loss(nn.Module):
    def __init__(self):
        super(MAPE_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # 避免除零
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 自定义 RMSE 损失函数
class RMSE_Loss(nn.Module):
    def __init__(self):
        super(RMSE_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

# 读取数据
data = pd.read_csv("../data/dataset.csv")
X_virtual = pd.read_csv("synthetic_data.csv")

# 数据分割
data['target_class'] = pd.qcut(data['Cs'], q=10, labels=False)
X = data.drop(['Cs', 'target_class'], axis=1).values
y = data['Cs'].values
stratify_column = data['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

# 数据标准化
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_virtual_scaled = scaler.transform(X_virtual)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

X_virtual_tensor = torch.tensor(X_virtual_scaled, dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建神经网络
from function import metrics_to_dataframe


class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 12)
        self.layer2 = nn.Linear(12, 90)
        self.layer3 = nn.Linear(90, 90)
        self.layer4 = nn.Linear(90, 60)
        self.layer5 = nn.Linear(60, 70)
        self.layer6 = nn.Linear(70, 30)
        self.output = nn.Linear(30, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = self.output(x)
        return x


# 初始化模型
input_dim = X_train_scaled.shape[1]
model = ANN(input_dim)

# 选择损失函数 (可以选择 MAPE 或 RMSE)
loss_function = MAPE_Loss()
# loss_function = RMSE_Loss()  # 或者使用 MAPE_Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 重新定义模型结构
model = ANN(input_dim)  # 使用相同的模型结构

# 加载模型参数，确保使用 weights_only=True 来提高安全性
model.load_state_dict(torch.load("ann_model.pth", weights_only=True))  # 加载模型参数

# 评估模型
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy()
    y_test_pred = model(X_test_tensor).numpy()
    y_virtual_pred = model(X_virtual_tensor).numpy()

# 保存合成数据
X_virtual['Cs'] = y_virtual_pred
X_virtual.to_csv('synthetic_data_with_predictions.csv', index=False)

# 计算并显示评估指标
ann_metrics = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, 'ANN')
Print(ann_metrics)
