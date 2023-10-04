import torch
import torch.nn as nn
import numpy as np
import pdb
torch.manual_seed(42)  #


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()  # Call the __init__ method of the parent class
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First (input) layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second (output) layer
    
    def forward(self, x):
        x = self.fc1(x)  # Pass input through the first layer
        x = self.relu(x)  # Apply ReLU activation function
        x = self.fc2(x)  # Pass through the second layer
        return x  # Return the output


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = MLP(hidden_dim, hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        return output
