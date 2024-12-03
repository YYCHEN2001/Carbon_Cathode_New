import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


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


# 自定义MAE损失函数
class MAE_Loss(nn.Module):
    def __init__(self):
        super(MAE_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred))