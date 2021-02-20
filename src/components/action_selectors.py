import torch as th
from torch.distributions import Categorical
import torch.nn.functional as F
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              args.epsilon_delay, decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              args.epsilon_delay, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class ModelEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.model_epsilon_start, args.model_epsilon_finish, args.model_epsilon_anneal_time,
                                              args.model_epsilon_delay, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)
        epsilon = 0.0 if test_mode else self.epsilon

        #print(f"t_env: {t_env}, test_mode: {test_mode}, epsilon:{self.epsilon}")
        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["model_epsilon_greedy"] = ModelEpsilonGreedyActionSelector

class MuZeroActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.ucb_temperature_start, args.ucb_temperature_finish, args.ucb_temperature_anneal_time,
                                              args.ucb_temperature_delay, decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.greedy_epsilon = 1e-3

    def select_action(self, agent_inputs, avail_actions, t_env, greedy=False):
        masked_policies = agent_inputs.clone()
        if greedy:
            masked_policies[avail_actions == 0.0] = -float("inf")
            self.epsilon = self.greedy_epsilon
        else:
            self.epsilon = self.schedule.eval(t_env)

        masked_policies = F.gumbel_softmax(masked_policies, dim=-1, tau=self.epsilon)
        picked_actions = Categorical(masked_policies).sample().long()
        return picked_actions

REGISTRY["muzero"] = MuZeroActionSelector
