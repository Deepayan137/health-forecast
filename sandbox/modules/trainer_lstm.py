import os
from matplotlib import pyplot as plt
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb

from .base_trainer import BaseTrainer

class LSTMTrainer(BaseTrainer):
    
    def compute_loss(self, X_batch, labels):
        N, T, D = X_batch.shape
        loss = 0
        output_sequence = torch.zeros((N, T-self.seed, D)).float().to(X_batch.device)
        TEACHER_FORCING_RATIO = 1.
        X_in = X_batch[:, :self.seed, :]
        for i in range(T-(self.seed+1)):
            y_in = X_batch[:, i+self.seed, :]
            output = self.model(X_in)
            output_sequence[:, i, :] = output[:, -1, :]
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
            if use_teacher_forcing:
                X_in = torch.cat((X_in[:, 1:, :], y_in.unsqueeze(1)), dim=1)
            else:
                X_in = torch.cat((X_in[:, 1:, :], output[:, -1, :].unsqueeze(1)), dim=1)

        loss = self.criterion(output_sequence, X_batch[:, self.seed:])
        # logits = self.cls_model(torch.cat([X_batch[:, :self.seed, :], output_sequence], dim=1))
        logits = self.cls_model(X_batch)
        pdb.set_trace()
        cls_loss = self.cls_criterion(logits, labels)
        accuracy = self.compute_accuracy(logits, labels)
        return loss, cls_loss, accuracy

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
        y_pred = torch.cat([X_batch.to(self.device), y_pred], dim=1)
        return y_pred

    def train_one_epoch(self, train_loader, do_jitter=False, do_mixup=False, pretrain=False):
        self.model.train()  # Switch to train mode
        epoch_loss, epoch_cls_loss = 0., 0.
        epoch_accuracy = 0.
        for X_train, labels in (train_loader):
            X_train = X_train.to(self.device)
            labels = labels.to(self.device)
            if do_jitter:
                if not pretrain:
                    X_train = self.add_jitter(X_train)
            if do_mixup:
                if not pretrain:
                    X_train = self.mixup_data(X_train)

            self.optimizer.zero_grad()  # Clear the gradients
            loss, cls_loss, accuracy = self.compute_loss(X_train, labels)
            combined_loss = 0.1*loss + 0.9*cls_loss
            combined_loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_accuracy += accuracy
        return epoch_loss / len(train_loader), epoch_cls_loss / len(train_loader), epoch_accuracy / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()  # Switch to evaluation mode
        loss, cls_loss = 0., 0.
        with torch.no_grad():
            for X_val, labels in val_loader:
                X_val = X_val.to(self.device)
                labels = labels.to(self.device)
                seed_input = X_val[:, :self.seed]
                y_pred = self.generate_predictions(seed_input, num_generations=X_val.shape[1]-self.seed)
                loss  += self.criterion(y_pred[:, -1, 0], X_val[:, -1, 0])
                logits = self.cls_model(y_pred)
                cls_loss += self.cls_criterion(logits, labels)
                accuracy = self.compute_accuracy(logits, labels)
        return loss/len(val_loader), cls_loss/len(val_loader), accuracy/len(val_loader)
