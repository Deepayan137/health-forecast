import os
from matplotlib import pyplot as plt
import numpy as np

import torch
import pdb

from .base_trainer import BaseTrainer

class Seq2SeqTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, seed, trial, 
        tf_ratio=1.0, tf_decay=0.999):
        super().__init__(opt, model, optimizer, criterion, seed, trial)
        self.opt = opt
        self.tf_ratio = tf_ratio
        self.tf_decay = tf_decay
    
    def generate_predictions(self, X_batch, model=None, num_generations=148):
        if not model:
            model = self.model
        X_batch = X_batch.to(self.device)
        return model(X_batch)
    
    def train_one_epoch(self, train_loader, do_jitter=False, do_mixup=False, pretrain=False):
        self.model.train()  # Switch to train mode
        epoch_loss = 0
        for X_train, y_train in (train_loader):
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            if do_jitter:
                if not pretrain:
                    X_train = self.add_jitter(X_train)
            if do_mixup:
                if not pretrain:
                    X_train, y_train = self.mixup_data(X_train, y_train)

            self.optimizer.zero_grad()  # Clear the gradients
            output = self.model(X_train, y_train, self.tf_ratio)
            loss = self.criterion(output, y_train)
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            epoch_loss += loss.item()
        if self.tf_ratio:
            self.tf_ratio *= self.tf_decay
        return epoch_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()  # Switch to evaluation mode
        loss = 0.
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                y_pred = self.generate_predictions(X_val)
                loss  += self.criterion(y_pred[:, -1, 0], y_val[:, -1, 0])
        return loss/len(val_loader)