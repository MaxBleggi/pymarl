import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueModel(nn.Module):
    """
    Estimate n-step discounted return
    """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, xt):
        xt = F.relu(self.fc1(xt))
        xt = self.fc2(xt)
        return xt