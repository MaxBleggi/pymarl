import numpy as np
import random

class Edge():
    """
    Maintains traversal stats between parent and child nodes
    """

    def __init__(self):
        self.count = 0
        self.value = 0
        self.prior = 0
        self.reward = 0


class Node():

    def __init__(self, name, action_space):
        self.name = name
        # the number of agents and actions per agent for the scenario
        self.action_space = action_space
        self.n_agents, self.n_actions = self.action_space

        # each possible joint action is recorded as a separate child
        # potential size is n_actions ** n_agents
        # a child is added when this node is expanded
        self.children = {}
        self.edges = self.init_edges()
        self.t = 0 # timestep represented by this node
        self.batch = None # this is a tuple of tensors representing the environment state, available actions and termination status
        self.count = 0

    def init_edges(self):
        edges = []
        for i in range(self.n_agents):
            edges.append([Edge() for j in range(self.n_actions)])
        return edges

    def expanded(self):
        return len(self.children) > 0

    def visit(self):
        self.count += 1

    def add_child(self, action, child):
        self.children[action] = child

    def __str__(self):
        return f"name: {self.name}, t: {self.t}, count: {self.count}"
