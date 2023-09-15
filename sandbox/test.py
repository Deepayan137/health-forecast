import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import *
import pdb
# Read CSV function
def read_csv(csv_path):
    df = pd.read_csv(csv_path, delimiter=',')
    df['Time'] = df['Time'].astype(int)
    data = np.zeros((100, 169, 1))
    labels = []
    for i, id in enumerate(df['Id'].unique()):
        patient_data = df[df['Id'] == id]
        if i < 50:
            labels.append(0)
        else:
            labels.append(1)
        data[i, patient_data['Time'].values] = patient_data[['V']].values
    return data, labels

# Load data
train_data, train_labels = read_csv('sandbox/data/phds/pred_training_1.csv')
val_data, val_labels = read_csv('sandbox/data/phds/pred_validation_1.csv')

# Convert to PyTorch tensors
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

# LSTM Classifier with Average Pooling
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.avg_pool = nn.AvgPool1d(kernel_size=169)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        pooled_out = self.avg_pool(lstm_out).squeeze(-1)
        output = self.classifier(pooled_out)
        return output.squeeze()

# Model, Loss, Optimizer
model = LSTMClassifier(input_dim=1, hidden_dim=64).cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dataloaders
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Training...")
num_epochs = 100
progress_bar = trange(num_epochs, desc="Epoch")

for epoch in progress_bar:
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / num_batches
    progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")

print("Training complete...Evaluating...")
# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.cuda()
        logits = model(X_batch)
        preds = torch.sigmoid(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())

# Calculate accuracy
predicted_labels = np.round(all_preds)
accuracy = accuracy_score(all_labels, predicted_labels)
print(accuracy)
