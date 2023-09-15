import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset for next-step prediction.
    """
    def __init__(self, data, labels, model_name='lstm', seed=21):
        """
        Args:
        data (numpy.array): Numpy array of time series data of shape (N, T, D)
        where N is the number of samples, T is the sequence length, and D is the
        dimensionality of the data at each time step.
        """
        self.data, self.labels = torch.tensor(data, dtype=torch.float32),\
        torch.tensor(labels, dtype=torch.float32)
        self.model_name = model_name
        self.seed = seed
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
        input_data = self.data[idx]
        label = self.labels[idx]
        # if self.model_name != "lstm":
        #     # Label is the last time step
        #     seed = self.seed
        #     input_data = self.data[idx, :seed, :]
        #     label = self.data[idx, seed:, :]
        #     return input_data, label
        return input_data, label
