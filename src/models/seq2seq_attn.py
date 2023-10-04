import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
torch.manual_seed(42)  #

from .seq2seq import Encoder

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # Transpose and reshape decoder_hidden
        decoder_hidden = decoder_hidden.permute(1, 2, 0)
        attn_weights = torch.bmm(encoder_outputs, decoder_hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)

        # Weighted sum of encoder_outputs (context) with attention weights
        weighted_sum = torch.bmm(soft_attn_weights.transpose(1, 2), encoder_outputs)

        return weighted_sum, soft_attn_weights

class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(DecoderWithAttention, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, x, hidden, encoder_outputs):
        weighted_sum, attn_weights = self.attention(encoder_outputs, hidden[0])
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.linear(lstm_out + weighted_sum)  # Add the attention weighted sum to the output
        return output, hidden, attn_weights

class Seq2SeqAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, target_len=148):
        super(Seq2SeqAttn, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = DecoderWithAttention(output_dim, hidden_dim)
        self.target_len = target_len

    def forward(self, src, trg=None, tf_ratio=None):
        # trg is None during inference
        N, _, D = src.shape
        encoder_outputs, hidden = self.encoder(src)
        if trg is None:  # inference mode
            trg = torch.zeros((N, self.target_len, D)).float().to(src.device)  # initialize target tensor
        output_sequence = torch.zeros_like(trg)
        input_to_decoder = trg[:, 0, :].unsqueeze(1)
        for t in range(1, self.target_len):
            output, hidden, _ = self.decoder(input_to_decoder, hidden, encoder_outputs)
            output_sequence[:, t, :] = output[:, -1, :]
            teacher_force = torch.rand(1).item() < tf_ratio if tf_ratio is not None else False
            input_to_decoder = trg[:, t, :].unsqueeze(1) if teacher_force and trg is not None else output
        return output_sequence
