from torch import nn
from torch_geometric.nn import GCNConv
import torch

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(GCNConv(hidden_dims[-1], output_dim))

        # eg. hidden_dims = [16, 8]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)
        x = self.layers[-1](x, edge_index)
        return x
