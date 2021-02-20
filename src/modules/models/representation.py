import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationModel(nn.Module):
    """
    Model to convert initial state into initial recurrent hidden state used by DynamicsModel
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, xt):
        xt, ht_ct = self.rnn(xt)
        xt = xt[:, -1, :] # select last timestep
        xt = F.relu(self.fc1(xt))
        return xt