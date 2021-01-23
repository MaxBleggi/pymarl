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
        self.f1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, at, ht_ct):
        xt = F.relu(self.f1(at)) # action embedding
        ht, ct = self.rnn(xt, ht_ct)

        return ht, ct

    def init_hidden(self, batch_size, device):
        ht = torch.zeros(batch_size, self.hidden_size).to(device)
        ct = torch.zeros(batch_size, self.hidden_size).to(device)
        return (ht, ct)