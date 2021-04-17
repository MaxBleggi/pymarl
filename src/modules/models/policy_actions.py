import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyActionsModel(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.s1 = state_size

        # policy head
        self.p_fc1 = nn.Linear(self.s1, action_size)

        # action head
        self.aa_fc1 = nn.Linear(self.s1, action_size)

    def forward(self, state):
        xt = state

        pt = self.p_fc1(xt)  # policy
        aa = self.aa_fc1(xt)  # avail actions

        return pt, aa


class PADynamicsModel(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.s1 = hidden_size
        self.input_size = state_size + action_size

        # state-action embedding
        self.sa_fc1 = nn.Linear(self.input_size, self.s1)

        # rnn
        self.rnn1 = nn.LSTMCell(self.s1, self.s1)

        # representation head
        self.s_fc1 = nn.Linear(self.s1, state_size)

    def forward(self, st, at, ht_ct):

        ht1, ct1 = ht_ct

        # embedding
        xt = torch.cat((st, at), dim=-1)
        xt = F.relu(self.sa_fc1(xt))

        # rnn
        ht1, ct1 = self.rnn1(xt, (ht1, ct1))

        # heads
        st = F.relu(self.s_fc1(ht1))  # state

        return st, (ht1, ct1)

    def init_hidden(self, batch_size, device):
        ht1 = torch.zeros(batch_size, self.s1).to(device)
        ct1 = torch.zeros(batch_size, self.s1).to(device)

        return (ht1, ct1)

