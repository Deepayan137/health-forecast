import os
from matplotlib import pyplot as plt
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb

from .base_trainer import BaseTrainer

class BiasedLSTMTrainer(BaseTrainer):
    
    def compute_loss(self, X_batch, labels):
        N, T, D = X_batch.shape
        loss = 0
        labels = labels.to(self.device)
        # Duplicate the labels
        expanded_labels = labels.unsqueeze(1).expand(-1, T).unsqueeze(2)
        output_sequence = torch.zeros((N, T-self.seed, 1)).float().to(X_batch.device)
        
        TEACHER_FORCING_RATIO = 1.
        
        X_in = X_batch[:, :self.seed, :]
        labels_in = expanded_labels[:, :self.seed, :]
        X_in = torch.cat((X_in, labels_in), dim=2)
        for i in range(T-self.seed):
            # Generate output using the model
            output = self.model(X_in)
            output_sequence[:, i, :] = output[:, -1, :]
            
            # Teacher forcing decision
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
            
            # Get the actual next value or predicted value based on teacher forcing
            next_input = X_batch[:, i+self.seed, :] if use_teacher_forcing else output[:, -1, 0].unsqueeze(1)
            
            # Prepare input for the next timestep

            X_in = torch.cat((X_in[:, 1:, 0], next_input), dim=1).unsqueeze(2)
            labels_in = expanded_labels[:, i+1:self.seed+i+1, :]
            X_in = torch.cat((X_in, labels_in), dim=2)
        
        # Compute the loss
        loss = self.criterion(output_sequence, X_batch[:, self.seed:])
        return loss


    def generate_val_predictions(self, X_batch, labels, model=None, num_generations=None):
        current_input = X_batch.to(self.device)
        labels = labels.to(self.device).unsqueeze(1).expand(-1, current_input.shape[1]).unsqueeze(2)
        current_input = torch.cat((current_input, labels), dim=2)
        predictions = []

        with torch.no_grad():
            for _ in range(num_generations):
                output = self.model(current_input)
                predictions.append(output[:, -1, :])
                # Remove the oldest observation and append latest output to current_input
                current_input = torch.cat((current_input[:, 1:, 0].unsqueeze(-1), output[:, -1].unsqueeze(-1)), dim=1)
                # Append the label to the input at every time step
                current_input = torch.cat((current_input, labels), dim=2)

        y_pred = torch.stack(predictions, dim=1)
        y_pred = torch.cat([X_batch.to(self.device), y_pred], dim=1)
        return y_pred


    def generate_test_predictions(self, X_seed, labels, model=None, num_generations=None):
        model.eval()  # Set the model to evaluation mode
        seed = X_seed.shape[1]
        # Move the seed input and labels to the device
        X_seed = X_seed.to(self.device)
        labels = labels.to(self.device).unsqueeze(1).expand(-1, num_generations).unsqueeze(2)
        
        predictions = []
        with torch.no_grad():
            for i in range(num_generations):
                if i < seed:
                    current_input = X_seed[:, i:i+1, :]  # Take the corresponding input from X_seed
                else:
                    current_input = predictions[-1]  # Take the last predicted output
                
                current_input = torch.cat((current_input, labels[:, i:i+1, :]), dim=2)  # Append the label
                
                output = model(current_input)  # Generate output
                predictions.append(output[:, -1, :].unsqueeze(1))  # Store the last output
        pdb.set_trace()
        # Combine all the predictions
        y_pred = torch.cat(predictions, dim=1)
        return y_pred


    def train_one_epoch(self, train_loader, do_jitter=False, do_mixup=False, pretrain=False):
        self.model.train()  # Switch to train mode
        epoch_loss = 0.
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
            loss = self.compute_loss(X_train, labels)
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()  # Switch to evaluation mode
        loss = 0.
        with torch.no_grad():
            for X_val, labels in val_loader:
                X_val = X_val.to(self.device)
                labels = labels.to(self.device)
                seed_input = X_val[:, :self.seed]
                y_pred = self.generate_val_predictions(seed_input, labels, num_generations=X_val.shape[1]-self.seed)
                loss  += self.criterion(y_pred[:, -1, 0], X_val[:, -1, 0])
        return loss/len(val_loader)
