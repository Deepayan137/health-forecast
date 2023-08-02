import torch
import torch.nn as nn
import numpy as np
import pdb
torch.manual_seed(42)  #

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.dropout = nn.Dropout(0.5)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        # lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        return output

