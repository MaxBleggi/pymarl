import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionsModel(nn.Module):
    """
    Generate available actions at t+1 from state representation and available actions at t
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, avt, ht):
        xt = torch.cat((ht, avt), dim=-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = self.fc3(xt) + avt # learn the residual
        return torch.clamp(xt, min=0, max=1)