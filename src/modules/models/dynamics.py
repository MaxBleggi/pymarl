import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):

    """
    Generates Reward, Termination and Value function outputs
    """

    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.input_size = state_size + action_size
        self.hidden_size = hidden_size

        self.s1 = self.hidden_size

        # state-action embedding
        self.sa_fc1 = nn.Linear(self.input_size, self.s1)

        # rnn
        self.rnn1 = nn.LSTMCell(self.s1, self.s1)

        # reward head
        self.r_fc1 = nn.Linear(self.s1, 1)

        # representation head
        self.s_fc1 = nn.Linear(self.s1, state_size)

        # value head
        self.v_fc1 = nn.Linear(self.s1, 1)

        # termination head
        self.t_fc1 = nn.Linear(self.s1, 1)


    def forward(self, st, at, ht_ct):

        # embedding
        xt = torch.cat((st, at), dim=-1)
        xt = F.relu(self.sa_fc1(xt))

        # rnn
        ht1, ct1 = ht_ct
        ht1, ct1 = self.rnn1(xt, (ht1, ct1))

        # heads
        st = F.relu(self.s_fc1(ht1))  # state
        rt = self.r_fc1(ht1)  # rewqrd
        vt = self.v_fc1(ht1)  # value
        tt = self.t_fc1(ht1)  # term

        return st, rt, vt, tt, (ht1, ct1)

    def init_hidden(self, batch_size, device):
        ht1 = torch.zeros(batch_size, self.s1).to(device)
        ct1 = torch.zeros(batch_size, self.s1).to(device)
        return (ht1, ct1)