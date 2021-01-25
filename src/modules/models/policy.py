import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    """
    Apporximate external policy
    """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, xt):
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = self.fc3(xt)

        return xt