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
        self.linear = MLP(hidden_dim, hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        output = self.linear(lstm_out)
        return output

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.5):
        super(EnhancedLSTMModel, self).__init__()
        
        # LSTM Layer with Dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # MLP with Batch Normalization and Dropout
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # We are not using the hidden state output here
        output = self.linear(lstm_out.squeeze())
        return output.unsqueeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Average pooling across the time dimension
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Pooling across the time dimension
        pooled_out = self.avg_pool(lstm_out.transpose(1, 2)).squeeze(2)
        
        output = self.classifier(pooled_out)
        return output.squeeze()