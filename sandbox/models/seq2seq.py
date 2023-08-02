import torch
import torch.nn as nn
import numpy as np
import pdb
torch.manual_seed(42)  #

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        encoder_outputs, hidden = self.lstm(x)
        return encoder_outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.linear(lstm_out)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim,
     output_dim, target_len=148):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(output_dim, hidden_dim)
        self.target_len = target_len

    def forward(self, src, trg=None, tf_ratio=None):
        # trg is None during inference
        N, _, D = src.shape
        _, hidden = self.encoder(src)
        if trg is None:  # inference mode
            trg = torch.zeros((N, self.target_len, D)).float().to(src.device)  # initialize target tensor
        output_sequence = torch.zeros_like(trg)
        input_to_decoder = trg[:, 0, :].unsqueeze(1)
        for t in range(1, self.target_len):
            output, hidden = self.decoder(input_to_decoder, hidden)
            output_sequence[:, t, :] = output[:, -1, :]
            teacher_force = torch.rand(1).item() < tf_ratio if tf_ratio is not None else False
            input_to_decoder = trg[:, t, :].unsqueeze(1) if teacher_force and trg is not None else output
        return output_sequence

if __name__ == "__main__":
    import pdb
    x = torch.randn((32, 21, 2))
    y = torch.randn((32, 148, 2))
    model = Seq2Seq(2, 128, 2)
    out = model(x)
