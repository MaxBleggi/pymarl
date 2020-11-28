import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    Model to predict r given s, a
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        xt = torch.cat((state, action), dim=-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = self.fc3(xt)
        return xt