import torch
from torch import nn


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