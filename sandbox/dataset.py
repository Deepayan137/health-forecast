import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset for next-step prediction.
    """
    def __init__(self, data):
        """
        Args:
        data (numpy.array): Numpy array of time series data of shape (N, T, D)
        where N is the number of samples, T is the sequence length, and D is the
        dimensionality of the data at each time step.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a sample for next-step prediction.

        Args:
        idx (int): Index of the sample to return.

        Returns:
        tuple: (input, label) where input is the time series data up to the
        second-to-last time step and label is the time series data at the last time step.
        """
        # Input is all but the last time step
        input_data = self.data[idx, :-1, :]
        # Label is the last time step
        label = self.data[idx, 1:, :]
        
        return input_data, label
