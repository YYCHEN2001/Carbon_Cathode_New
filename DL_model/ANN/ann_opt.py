import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 载入数据集并预处理
data = pd.read_csv("../../data/dataset.csv")

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


# 定义一个PyTorch模型，支持每一层的神经元数目可调
class ANNModel(nn.Module):
    def __init__(self, input_dim, layer_neurons):
        super(ANNModel, self).__init__()
        layers = []

        # 第一个隐藏层
        layers.append(nn.Linear(input_dim, layer_neurons[0]))
        layers.append(nn.ReLU())

        # 中间隐藏层
        for i in range(1, len(layer_neurons)):
            layers.append(nn.Linear(layer_neurons[i - 1], layer_neurons[i]))
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(layer_neurons[-1], 1))  # 假设是回归问题
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

base_neuron = 10  # 每层神经元的基数

# 定义目标函数用于Optuna
def objective(trial):
    # 超参数搜索空间
    n_layers = trial.suggest_int('n_layers', 4, 8)  # 隐藏层数
    layer_neurons = []

    # 为每一层定义神经元的数量（每层神经元数目是基数的整数倍）
    for i in range(n_layers):
        multiplier = trial.suggest_int(f'multiplier_l{i + 1}', 1, 13)  # 隐藏层神经元数目的倍数
        layer_neurons.append(base_neuron * multiplier)  # 每层神经元数目是基数的整数倍

    # 使用 suggest_float 代替 suggest_loguniform
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # 学习率

    # 创建模型
    model = ANNModel(input_dim=X_train_scaled.shape[1], layer_neurons=layer_neurons)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 使用5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # 将数据转化为PyTorch张量
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).view(-1, 1).to(device)

        # 训练模型
        model.to(device)
        model.train()
        for epoch in range(1000):  # 设定训练周期为100
            optimizer.zero_grad()
            y_pred = model(X_train_tensor)
            loss = criterion(y_pred, y_train_tensor)
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = mean_squared_error(y_val_tensor.cpu().numpy(), y_val_pred.cpu().numpy())
            scores.append(-val_loss)  # 负均方误差

    return np.mean(scores)  # 返回平均的负均方误差


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')  # 我们要最大化交叉验证得分
study.optimize(objective, n_trials=50)  # 搜索50次


print("Best trial:")
best_trial = study.best_trial
print(f"  Number of layers: {best_trial.params['n_layers']}")
for i in range(best_trial.params['n_layers']):
    # 计算每层神经元数目
    num_neurons = best_trial.params[f'multiplier_l{i+1}'] * base_neuron
    print(f"  Number of neurons in layer {i+1}: {num_neurons}")
print(f"  Learning rate: {best_trial.params['learning_rate']}")
