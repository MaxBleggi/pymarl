from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class MuZeroMAC:
    def __init__(self, args):
        self.args = args
        self.action_selector = action_REGISTRY[args.model_action_selector](args)

    # used by runners to generate real experience
    def select_actions(self, agent_outputs, avail_actions, t_env, test_mode=False):
        # Only select actions for the selected batch elements in bs
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, t_env, test_mode=test_mode)
        return chosen_actions

