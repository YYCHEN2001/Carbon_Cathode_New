import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from function import split_data
from torch_function import RMSE_Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv("../../data/dataset_reduced.csv")
X_train, X_test, y_train, y_test = split_data(data, 'Cs')

# Data standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)

# Define LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            lstm_dropout = dropout if self.num_layers > 1 and i < self.num_layers - 1 else 0
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True, dropout=lstm_dropout))

        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x, seq_lengths=None):
        h0 = [torch.zeros(1, x.size(0), hidden_size).to(x.device) for hidden_size in self.hidden_sizes]
        c0 = [torch.zeros(1, x.size(0), hidden_size).to(x.device) for hidden_size in self.hidden_sizes]

        out = x
        for i, lstm in enumerate(self.lstm_layers):
            if seq_lengths is not None:
                packed_input = nn.utils.rnn.pack_padded_sequence(out, seq_lengths, batch_first=True, enforce_sorted=False)
                packed_output, (h, c) = lstm(packed_input, (h0[i], c0[i]))
                out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                out, (h, c) = lstm(out, (h0[i], c0[i]))

        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out

# Reshape input for LSTM (batch_size, sequence_length, input_size)
def reshape_for_lstm(X):
    return X.unsqueeze(1)  # Add a sequence length dimension of 1

# Define objective function
def objective(trial):
    hidden_sizes = [trial.suggest_int('n_units_l{}'.format(i), 4, 128) for i in range(trial.suggest_int('n_layers', 2, 6))]
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 0.2, log=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_index, val_index in kf.split(X_train_tensor):
        X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
        y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]

        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = LSTMRegressor(input_size, hidden_sizes, output_size, dropout).to(device)
        criterion = RMSE_Loss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        num_epochs = 300
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = reshape_for_lstm(X_batch)

                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = reshape_for_lstm(X_batch)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

    return np.mean(val_losses)

input_size = X_train_scaled.shape[1]
output_size = 1

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Output best hyperparameters
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))