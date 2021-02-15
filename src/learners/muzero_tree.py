import torch
import torch.nn.functional as F

class TreeState():
    # this is a set of tensors representing the dynimacis model hidden state, available actions and termination status
    def __init__(self, ht, ct, avail_actions, term_signal):

        self.ht = ht
        self.ct = ct
        self.avail_actions = avail_actions
        self.term_signal = term_signal

    def totuple(self):
        return (self.ht, self.ct, self.avail_actions, self.term_signal)

class Node():

    def __init__(self, name, action_space, action=None, t=0, device='cpu'):
        self.name = name
        # the number of agents and actions per agent for the scenario
        self.action_space = action_space
        self.n_agents, self.n_actions = self.action_space
        self.device = device

        self.t = t  # timestep represented by this node
        self.children = {} # each possible joint action is recorded as a separate child
        self.state = None # TreeState
        self.priors = torch.zeros(self.action_space, device=device)
        self.child_visits = torch.zeros(self.action_space, device=device)
        self.action_values = torch.zeros(self.action_space, device=device)
        self.reward = 0
        self.value = 0
        self.count = 0
        self.terminal = False

    def visit(self):
        self.count += 1

    def add_child(self, action, child):
        self.children[action] = child

    def update(self, Q, V, R, terminal, state):
        self.state = state
        self.priors = F.softmax(Q[-1], dim=-1)
        self.action_values = Q[-1]
        self.value = V[-1].item()
        self.reward = R[-1].item()
        self.terminal = terminal[-1].item()

    def backup(self, action, G):
        self.child_visits.scatter_add_(1, torch.tensor([action], device=self.device).view(-1, 1), torch.ones(self.action_space, device=self.device))
        self.count += 1


    def __str__(self):
        return f"name: {self.name}, t: {self.t}, count: {self.count}"
