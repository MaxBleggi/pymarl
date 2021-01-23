import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):
    """
    Generate reward and termination signal given the action history and an initial hidden state generated
    by the Representation model
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)

    def forward(self, at, ht_ct):
        ht, ct = self.rnn(at, ht_ct)

        return ht, ct

    def init_hidden(self, batch_size, device):
        ht = torch.zeros(batch_size, self.hidden_size).to(device)
        ct = torch.zeros(batch_size, self.hidden_size).to(device)
        return (ht, ct)