import torch
import torch.nn as nn
import torch.nn.functional as F

class TermModel(nn.Module):
    """
    Generate term signal from state representation
    """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, output_size)
        self.d1 = nn.Dropout(p=0.5)

    def forward(self, xt):
        xt = F.relu(self.d1(self.fc1(xt)))

        return xt