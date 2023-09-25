import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import *
import pdb
import pickle

from sandbox.data_utils import read_csv_with_missing_val, read_csv, CustomScaler
from sandbox.models import LSTMModel, LSTMClassifier
from sandbox.impute import impute_missing_values
# Load data
import pandas as pd
import numpy as np



train_data, train_labels = read_csv_with_missing_val("sandbox/data/phds/training_1.csv")
# train_data, saits_0, saits_1 = impute_missing_values(train_data, train_labels)
train_data = np.load("train_imputed.npy")
val_data, val_labels = read_csv_with_missing_val("sandbox/data/phds/validation_1.csv")
# val_data, _, _ = impute_missing_values(val_data, val_labels)
val_data = np.load("val_imputed.npy")
scaler = CustomScaler()
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)
# Convert to PyTorch tensors
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)


# Model, Loss, Optimizer
model = LSTMClassifier(input_dim=1, hidden_dim=64).cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Dataloaders
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Training...")
num_epochs = 100
progress_bar = trange(num_epochs, desc="Epoch")

last_loss = float('inf')  # Initialize with a high value

for epoch in progress_bar:
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = len(train_loader)
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Only backpropagate and update if the current loss is less than the last loss
        if loss.item() < last_loss:
            preds = torch.sigmoid(logits)
            preds = torch.round(preds).cpu().detach().numpy()
            tr_acc = accuracy_score(y_batch.cpu().detach().numpy(), preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_loss += loss.item()
        epoch_accuracy += tr_acc.item()
    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_accuracy = epoch_accuracy / num_batches
    last_loss = avg_epoch_loss  # Update the last_loss value
    
    progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")
    print(f"Epoch:{epoch}...Accuracy:{avg_epoch_accuracy}")
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
