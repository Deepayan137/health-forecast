import os
from matplotlib import pyplot as plt
import numpy as np

import torch
import pdb

from .base_trainer import BaseTrainer

class LSTMTrainer(BaseTrainer):
    
    def compute_loss(self, X_batch):
        N, T, D = X_batch.shape
        loss = 0
        output_sequence = torch.zeros((N, T-self.seed, D)).float().to(X_batch.device)
        for i in range(T-(self.seed+1)):
            X_in = X_batch[:, i:i+self.seed, :]
            y_in = X_batch[:, i+self.seed, :]
            output = self.model(X_in)
            output_sequence[:, i, :] = output[:, -1, :]
        if str(self.criterion) == "MaskedMSELoss":
            mask = torch.where(X_batch>0.0, 1.0, 0.0).to(X_batch.device)
            loss = self.criterion(output_sequence, X_batch[:, self.seed:], mask[:, self.seed:].bool())
        else:
            loss = self.criterion(output_sequence, X_batch[:, self.seed:])
        return loss

    def generate_predictions(self, X_batch, model=None, num_generations=None):
        if not model:
            model = self.model
        current_input = X_batch.to(self.device)
        predictions = []
        with torch.no_grad():
            for _ in range(num_generations):
                output = self.model(current_input)
                predictions.append(output[:, -1, :])
                current_input = torch.cat((current_input[:, 1:, :], output[:, -1, :].unsqueeze(1)), dim=1)
        y_pred = torch.stack(predictions, dim=1)
        return y_pred

    def train_one_epoch(self, train_loader, do_jitter=False, do_mixup=False):
        self.model.train()  # Switch to train mode
        epoch_loss = 0
        for X_train in (train_loader):
            X_train = X_train.to(self.device)
            if do_jitter:
                X_train = self.add_jitter(X_train)
            if do_mixup:
                X_train = self.mixup_data(X_train)

            self.optimizer.zero_grad()  # Clear the gradients
            loss = self.compute_loss(X_train)
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()  # Switch to evaluation mode
        loss = 0.
        with torch.no_grad():
            for X_val in val_loader:
                X_val = X_val.to(self.device)
                seed_input = X_val[:, :self.seed]
                y_pred = self.generate_predictions(seed_input, num_generations=X_val.shape[1]-self.seed)
                if str(self.criterion) == "MaskedMSELoss":
                    mask = torch.where(X_val>0.0, 1.0, 0.0).to(X_val.device)
                    loss += self.criterion(y_pred, X_val[:, self.seed:], mask[:, self.seed:].bool())
                else:
                    loss  += self.criterion(y_pred[:, -1, 0], X_val[:, -1, 0])
        return loss/len(val_loader)
