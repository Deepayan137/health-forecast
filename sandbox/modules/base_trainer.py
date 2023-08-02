import os
from matplotlib import pyplot as plt
import numpy as np

import torch
import pdb


class BaseTrainer:
    def __init__(self, model, optimizer, criterion, seed, trial):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trial = trial
        # move the model to the device
        self.model.to(self.device)
        self.seed = seed
    
    def generate_predictions(self, X_batch, model=None, num_generations=None):
        raise NotImplementedError()

    def train_one_epoch(self, train_loader, do_jitter=False, do_mixup=False):
        raise NotImplementedError()

    def evaluate(self, val_loader):
        raise NotImplementedError()

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

    def mixup_data(self, x, y=None, alpha=1.0):
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
        if y is not None:
            y_mix = lam * y + (1 - lam) * y[idx]
            return x_mix, y_mix
        
        return x_mix

    def plot_losses(self, df, path):
        plt.plot(range(len(df)), df['train_loss'], color='red', label='train_loss')
        plt.plot(range(len(df)), df['val_loss'], color='blue', label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f"{path}/loss_{self.trial}.png")
        plt.close()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved at {path}. It\'s ready to shine another day, darling!')

    def load(self, path):

        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f'Model loaded from {path}. Back on the runway and ready to serve, darling!')
        return self.model