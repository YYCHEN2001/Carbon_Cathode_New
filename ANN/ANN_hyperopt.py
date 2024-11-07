import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope


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

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 定义搜索空间
space = {
    'num_layers': scope.int(hp.quniform('num_layers', 1, 5, 1)),  # 隐藏层的层数范围为 1 到 5 层
    'hidden_units': hp.choice('hidden_units', [
        [scope.int(hp.quniform(f'units_layer_{i}', 10, 120, 10)) for i in range(5)]
    ]),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'batch_size': scope.int(hp.quniform('batch_size', 16, 64, 16)),
}


# 定义动态模型类，允许层数和节点数的可变性
class DynamicANN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_units):
        super(DynamicANN, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for i in range(num_layers):
            self.layers.append(nn.Linear(prev_dim, hidden_units[i]))
            prev_dim = hidden_units[i]
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output(x)
        return x


# 定义目标函数
def objective(params):
    # 解包参数
    num_layers = params['num_layers']
    hidden_units = params['hidden_units'][:num_layers]  # 只取需要的层数的单元数
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']

    # 重新创建 DataLoader 以适应新的 batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数和优化器
    model = DynamicANN(X_train_scaled.shape[1], num_layers, hidden_units)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = RMSE_Loss()

    # 训练模型
    num_epochs = 200  # 可适当调整以加快搜索
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_tensor).numpy()
        rmse = np.sqrt(np.mean((y_test - y_test_pred.squeeze()) ** 2))

    # 返回验证 RMSE 作为损失
    return rmse


# 使用 TPE 算法进行超参数优化
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

# 输出最佳参数
print("Best parameters:", best)

# 使用最佳参数重新训练和保存模型
best_num_layers = int(best['num_layers'])  # 转换为整数
best_hidden_units = space['hidden_units'][int(best['hidden_units'])][:best_num_layers]  # 转换为整数并获取合适层数的单元数
best_learning_rate = float(best['learning_rate'])  # 转换为浮点数
best_batch_size = int(best['batch_size'])  # 转换为整数

# 创建和保存最终模型
final_model = DynamicANN(X_train_scaled.shape[1], best_num_layers, best_hidden_units)
optimizer = optim.Adam(final_model.parameters(), lr=best_learning_rate)
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
torch.save(final_model.state_dict(), "best_hyperopt_ann_model.pth")
