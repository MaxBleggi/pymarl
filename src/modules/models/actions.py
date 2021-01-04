import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionsModel(nn.Module):
    """
    Generate available actions from state representation
    """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, xt):
        xt = F.relu(self.fc1(xt))
        return xt