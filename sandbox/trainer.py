import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pdb

class Trainer:
    def __init__(self, model, optimizer, criterion, trial):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trial = trial
        # move the model to the device
        self.model.to(self.device)
    
    def train_one_epoch(self, train_loader, do_jitter=False, do_mixup=False):
        self.model.train()  # Switch to train mode
        epoch_loss = 0

        for X_train, y_train in (train_loader):
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            if do_jitter:
                X_train = self.add_jitter(X_train)
            if do_mixup:
                X_train, y_train = self.mixup_data(X_train, y_train)

            self.optimizer.zero_grad()  # Clear the gradients
            outputs = self.model(X_train)  # Forward pass
            loss = self.criterion(outputs.squeeze(), y_train.squeeze())  # Calculate loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)

    def add_jitter(self, X, jitter_factor=0.05):
        """
        Adds small random noise to the data.
    
        Args:
        X (torch.Tensor): The input data.
        jitter_factor (float): The factor of random noise.
    
        Returns:
        torch.Tensor: The data with added jitter.
        """
        _, seq_len, num_features = X.size()
        jitter = torch.randn(seq_len, num_features).to(self.device) * jitter_factor
        return X + jitter

    def mixup_data(self, x, y, alpha=1.0):
        """
        Applies mixup augmentation to a batch of time series data.

        Args:
        x (torch.Tensor): The input data. Shape should be (batch_size, sequence_length, num_features).
        y (torch.Tensor): The labels corresponding to x.
        alpha (float): The parameter for the beta distribution.

        Returns:
        torch.Tensor, torch.Tensor: The augmented data and corresponding labels.
        """
        # Get the size of the data
        batch_size, seq_len, num_features = x.size()

        # Sample a value for lambda from a beta distribution
        lam = np.random.beta(alpha, alpha)

        # Make sure that lambda + (1 - lambda) = 1
        lam = max(lam, 1 - lam)

        # Pick a random batch of data for the mixup
        idx = torch.randperm(batch_size).to(self.device)

        # Perform the mixup
        x_mix = lam * x + (1 - lam) * x[idx]
        y_mix = lam * y + (1 - lam) * y[idx]

        return x_mix, y_mix

    def evaluate(self, val_loader):
        self.model.eval()  # Switch to evaluation mode
        epoch_loss = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                outputs = self.model(X_val)
                # loss = self.criterion(outputs.squeeze(), y_val.squeeze())
                loss = self.criterion(outputs.squeeze(), y_val.squeeze())
                epoch_loss += loss.item()
        
        return epoch_loss / len(val_loader)
    
    def generate_predictions(self, model, seed_input, num_generations):
        model.eval()
        current_input = seed_input[:, 0].unsqueeze(1)
        predictions = []

        with torch.no_grad():
            for i in range(num_generations):
                output = model(current_input)
                if i < seed_input.shape[1]-1:
                    current_input = seed_input[:, i+1, :].unsqueeze(1)
                else:
                    current_input = output
                    predictions.append(output)
                # current_input = torch.cat((current_input[:, 1:, :], output.unsqueeze(1)), dim=1
        predictions = torch.stack(predictions, dim=1).squeeze()
        final = torch.cat([seed_input, predictions], dim=1)
        return final

    def plot_losses(self, df):
        plt.plot(range(len(df)), df['train_loss'], color='red', label='train_loss')
        plt.plot(range(len(df)), df['val_loss'], color='blue', label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f"sandbox/saved/loss_{self.trial}.png")
        plt.close()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved at {path}. It\'s ready to shine another day, darling!')

    def load(self, path):

        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f'Model loaded from {path}. Back on the runway and ready to serve, darling!')
        return self.model