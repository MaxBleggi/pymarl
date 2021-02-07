import torch
import torch.nn.functional as F

class Node():

    def __init__(self, name, action_space, device='cpu'):
        self.name = name
        # the number of agents and actions per agent for the scenario
        self.action_space = action_space
        self.n_agents, self.n_actions = self.action_space

        # each possible joint action is recorded as a separate child
        # potential size is n_actions ** n_agents
        # a child is added when this node is expanded
        self.children = {}
        # self.edges = self.init_edges()
        self.t = 0 # timestep represented by this node
        self.state = None # this is a tuple of tensors representing the dynimacis model hidden state, available actions and termination status
        self.priors = torch.zeros(self.action_space, device=device)
        self.child_visits = torch.zeros(self.action_space, device=device)
        self.action_values = torch.zeros(self.action_space, device=device)
        self.reward = 0
        self.value = 0
        self.count = 0
        self.terminal = False
        self.device = device

    def expanded(self):
        return len(self.children) > 0

    def visit(self):
        self.count += 1

    def add_child(self, action, child):
        self.children[action] = child
        self.child_visits.scatter_add_(1, torch.tensor([action], device=self.device).view(-1, 1), torch.ones(self.action_space, device=self.device))

    def update(self, Q, V, R, terminal, state, t):
        self.state = state
        self.priors = F.softmax(Q[-1])
        self.action_values = Q[-1]
        self.value = V[-1].item()
        self.reward = R[-1].item()
        self.terminal = terminal[-1].item()
        self.t = t

    def __str__(self):
        return f"name: {self.name}, t: {self.t}, count: {self.count}"
