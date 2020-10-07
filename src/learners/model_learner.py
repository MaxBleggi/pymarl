import time
import torch
import torch.nn.functional as F

from modules.models.representation import RepresentationModel
from modules.models.dynamics import DynamicsModel
from modules.models.actions import ActionsModel
from modules.models.policy import PolicyModel

import numpy as np
import random
from components.episode_buffer import EpisodeBatch
from functools import partial
from envs import REGISTRY as env_REGISTRY
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.distributions import Categorical
from controllers import REGISTRY as mac_REGISTRY


class ModelLearner:
    def __init__(self, mac, scheme, groups, logger, args):

        self.target_mac = mac # this is the actual policy network which we learn to emulate
        self.model_mac = mac_REGISTRY[args.model_mac](scheme, groups, args)
        self.scheme = scheme
        self.logger = logger
        self.args = args
        self.device = self.args.device

        # used to get env metadata
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        # input/output dimensions
        self.actions_size = args.n_actions * args.n_agents
        self.state_size = args.state_shape - self.actions_size if args.env_args["state_last_action"] else args.state_shape
        self.reward_size = scheme["reward"]["vshape"][0]
        self.term_size = scheme["terminated"]["vshape"][0]

        # dynamics model
        dynamics_model_input_size = self.actions_size
        self.dynamics_model_output_size = self.reward_size + self.term_size
        self.dynamics_model = DynamicsModel(dynamics_model_input_size, self.dynamics_model_output_size,
                                       args.dynamics_model_hidden_dim)

        # representation model
        representation_model_input_size = self.state_size
        representation_model_output_size = args.dynamics_model_hidden_dim
        self.representation_model = RepresentationModel(representation_model_input_size, representation_model_output_size,
                                                   args.representation_model_hidden_dim)

        # actions model
        actions_model_input_size = args.dynamics_model_hidden_dim
        actions_model_output_size = self.actions_size
        self.actions_model = ActionsModel(actions_model_input_size, actions_model_output_size, args.actions_model_hidden_dim)

        # policy model
        policy_model_input_size = args.dynamics_model_hidden_dim
        policy_model_output_size = self.actions_size
        self.policy_model = PolicyModel(policy_model_input_size, policy_model_output_size, args.policy_model_hidden_dim)

        # optimizer
        self.params = list(self.dynamics_model.parameters()) + list(self.representation_model.parameters()) \
                      + list(self.actions_model.parameters()) + list(self.policy_model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.args.model_learning_rate)

        # logging stats
        self.train_loss, self.val_loss = 0, 0
        self.train_r_loss, self.val_r_loss = 0, 0
        self.train_T_loss, self.val_T_loss = 0, 0
        self.train_aa_loss, self.val_aa_loss = 0, 0
        self.train_p_loss, self.val_p_loss = 0, 0

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train_test_split(self, indices, test_ratio=0.1, shuffle=True):

        if shuffle:
            random.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(test_ratio * n))
        train_indices = range(n - n_test)
        test_indices = range(len(train_indices), n)

        return train_indices, test_indices

    def get_episode_vars(self, ep):

        if ep.device != self.device:
            ep.to(self.device)

        # per-agent quantities
        obs = ep["obs"][:, :-1, ...]  # observations
        aa = ep["avail_actions"][:, :-1, ...].float()  # available actions
        action = ep["actions_onehot"][:, :-1, ...]  # actions taken

        # flatten per-agent quantities
        nbatch, ntimesteps, _, _ = obs.size()
        obs = obs.view((nbatch, ntimesteps, -1))
        aa = aa.view((nbatch, ntimesteps, -1))
        action = action.view(nbatch, ntimesteps, -1)

        # state
        state = ep["state"][:, :-1, :]
        if self.args.env_args["state_last_action"]:
            state = state[:, :, :self.state_size]

        # reward
        reward = ep["reward"][:, :-1, :]

        # termination signal
        terminated = ep["terminated"][:, :-1].float()
        term_idx = torch.squeeze(terminated).max(0)[1].item()
        term_signal = torch.ones_like(terminated)
        term_signal[:, :term_idx, :] = 0

        # generate current policy outputs
        policy = torch.zeros_like(action)
        with torch.no_grad():
            self.target_mac.init_hidden(nbatch)
            for t in range(term_idx + 1):
                policy[:, t, :] = self.target_mac.forward(ep, t=t).flatten()

        # mask for active timesteps (except for term_signal which is always valid)
        mask = torch.ones_like(terminated)
        mask[:, term_idx + 1:, :] = 0

        obs *= mask
        aa *= mask
        action *= mask
        reward *= mask
        state *= mask
        policy *= mask

        return state, action, reward, term_signal, obs, aa, policy, mask

    def get_batch(self, episodes, batch_size, use_mask=False):
        bs = min(batch_size, len(episodes))
        batch = random.sample(episodes, bs)
        props = [torch.cat(t) for t in zip(*batch)]
        if use_mask:
            mask = props[-1]
            idx = int(mask.sum(1).max().item())
            props = [x[:, :idx, :] for x in props]
        return props

    def get_model_input_output(self, state, actions, reward, term_signal, obs, aa, policy, mask):

        # inputs
        s = state[:, :-1, :]  # state at time t
        a = actions[:, :-1, :]  # joint action at time t
        av = aa[:, 1:, :]  # available actions at t
        pt = policy[:, 1:, :] # raw policy outputs at t

        # outputs
        r = reward[:, :-1, :]  # reward at time t+1
        T = term_signal[:, :-1, :]  # terminated at t+1
        nav = aa[:, 1:, :] # available actions at t+1

        y = torch.cat((r, T, nav, pt), dim=-1)
        return s, a, y

    def run_model(self, state, actions, ht_ct=None):

        bs, steps, n_actions = actions.size()
        ht, ct = ht_ct if ht_ct else self.dynamics_model.init_hidden(bs, self.device)
        output_size = self.dynamics_model_output_size + 2 * self.actions_size
        yp = torch.zeros(bs, steps, output_size).to(self.device)

        for t in range(0, steps):

            if t == 0:
                st = state[:, t, :]
                ht = self.representation_model(st)

            at = actions[:, t, :]
            pt = self.policy_model(ht)

            # step forward
            yt, (ht, ct) = self.dynamics_model(at, (ht, ct))
            avt = self.actions_model(ht)
            yp[:, t, :] = torch.cat((yt, avt, pt), dim=-1)

        return yp, (ht, ct)

    def train_models(self, train_episodes, test_episodes):

        print(f"Training models ...")

        # model learning parameters
        batch_size = min(self.args.model_batch_size, len(test_episodes))
        log_epochs = self.args.model_log_epochs
        use_mask = False # learning a termination signal is easier with unmasked input

        train_loss = 0
        val_loss = 0

        for e in range(self.args.model_epochs):
            t_start = time.time()

            self.representation_model.train()
            self.dynamics_model.train()
            self.actions_model.train()

            # get data
            props = self.get_batch(train_episodes, batch_size, use_mask=use_mask)
            state, actions, y = self.get_model_input_output(*props)

            # make predictions
            yp, _ = self.run_model(state.to(self.device), actions.to(self.device))

            # gradient descent
            self.optimizer.zero_grad()
            loss_vector = F.mse_loss(yp, y.to(self.device), reduction='none')
            loss = loss_vector.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, self.args.model_grad_clip_norm)
            self.optimizer.step()

            # record losses
            self.train_loss = loss.item()
            self.train_r_loss = loss_vector[:, :, 0].mean()
            self.train_T_loss = loss_vector[:, :, 1].mean()
            self.train_aa_loss = loss_vector[:, :, 2:self.actions_size].mean()
            self.train_p_loss = loss_vector[:, :, 2 + self.actions_size:].mean()

            if (e + 1) % log_epochs == 0:

                self.representation_model.eval()
                self.dynamics_model.eval()
                self.actions_model.eval()

                with torch.no_grad():
                    props = self.get_batch(test_episodes, batch_size, use_mask=use_mask)
                    state, actions, y = self.get_model_input_output(*props)
                    yp, _ = self.run_model(state.to(self.device), actions.to(self.device))
                    loss_vector = F.mse_loss(yp, y.to(self.device), reduction='none')

                    self.val_loss = loss_vector.mean()
                    self.val_r_loss = loss_vector[:, :, 0].mean()
                    self.val_T_loss = loss_vector[:, :, 1].mean()
                    self.val_aa_loss = loss_vector[:, :, 2:self.actions_size].mean()
                    self.val_p_loss = loss_vector[:, :, 2 + self.actions_size:].mean()

                # report epoch losses
                t_step = (time.time() - t_start)
                print("Environment Model:")
                print(f"epoch: {e + 1:<3}   train loss: {self.train_loss:.5f}, val loss: {self.val_loss:.5f} time: {t_step:.2f} s")
                print(f" -- reward: train {self.train_r_loss:.5f} val {self.val_r_loss:.5f}")
                print(f" -- term: train {self.train_T_loss:.5f} val {self.val_T_loss:.5f}")
                print(f" -- avail_actions: train {self.train_aa_loss:.5f} val {self.val_aa_loss:.5f}")
                print(f" -- policy: train {self.train_p_loss:.5f} val {self.val_p_loss:.5f}")

                # self.logger.console_logger.info(f"Model training epoch {i}")

    def train(self, buffer, t_env):

        # generate training and test episode indices
        indices = list(range(0, buffer.episodes_in_buffer))
        train_indices, test_indices = self.train_test_split(indices, test_ratio=self.args.model_test_ratio, shuffle=True)

        # extract episodes
        train_episodes = [self.get_episode_vars(buffer[i]) for i in train_indices]
        test_episodes = [self.get_episode_vars(buffer[i]) for i in train_indices]

        self.train_models(train_episodes, test_episodes)

    # def generate_batch(self, buffer, batch_size, t_env):
    #
    #     self.representation_model.eval()
    #     self.dynamics_model.eval()
    #     self.actions_model.eval()
    #
    #     with torch.no_grad():
    #         # sample real starts from the replay buffer
    #         episodes = buffer.sample(batch_size)
    #         self.logger.console_logger.info(f"Generating {batch_size} model based episodes")
    #
    #         # create new episode batch for generated episodes
    #         scheme = buffer.scheme.copy()
    #         scheme.pop("filled", None)  # buffer scheme excluding filled key
    #         batch = partial(EpisodeBatch, scheme, buffer.groups, batch_size, buffer.max_seq_length,
    #                         preprocess=buffer.preprocess, device=self.device)()
    #
    #         # get real starting states for the batch
    #         state = episodes["state"][:, 0, :self.state_size].unsqueeze(1).to(self.device)
    #         avail_actions = episodes["avail_actions"][:, 0].unsqueeze(1).to(self.device)
    #         actions_onehot = torch.zeros_like(episodes["actions_onehot"][:, 0].view(batch_size, 1, -1)).to(self.device)
    #         term_signal = episodes["terminated"][:, 0].unsqueeze(1).float().to(self.device)
    #         terminated = (term_signal > 0)
    #         active_episodes = [i for i, finished in enumerate(terminated.flatten()) if not finished]
    #
    #         # initialise hidden states
    #         ht_ct = None # dynamics model hidden state
    #
    #         max_t = batch.max_seq_length - 1
    #         # generate episode sequence
    #         print(f"Collecting {self.args.model_rollout_batch_size} episodes from MODEL ENV using epsilon: {self.mac.action_selector.epsilon:.2f}")
    #         for t in range(max_t):
    #
    #             pre_transition_data = {
    #                 #"state": batch_state[active_episodes],
    #                 "avail_actions": avail_actions[active_episodes],
    #                 "obs": obs[active_episodes]
    #             }
    #             batch.update(pre_transition_data, bs=active_episodes, ts=t)
    #
    #             # choose actions following current policy
    #             actions = self.model_mac.select_actions(batch, t_ep=t, t_env=t_env, bs=active_episodes).unsqueeze(1)
    #
    #             batch.update({"actions": actions}, bs=active_episodes, ts=t)  # this will generate actions_onehot
    #             actions_onehot = batch["actions_onehot"][:, t, ...].view(batch_size, 1, -1)  # latest action
    #
    #             # generate next state, reward and termination signal
    #             output, s_ht_ct = self.run_model(state, actions_onehot, avail_actions, ht_ct)
    #             state = output[:, :, :self.dynamics_model.hidden_size]; idx = self.dynamics_model.hidden_size
    #             reward = output[:, :, idx:idx + self.reward_size]; idx += self.reward_size
    #             term_signal = output[:, :, idx:idx + self.term_size]
    #             avail_actions = output[:, :, idx: idx + self.actions_size]
    #
    #             # generate termination mask
    #             threshold = 0.9
    #             terminated = (term_signal > threshold)
    #
    #             # if this is the last timestep, terminate
    #             if t == max_t - 1:
    #                 terminated[active_episodes] = True
    #
    #             post_transition_data = {
    #                 "reward": reward[active_episodes],
    #                 "terminated": terminated[active_episodes]
    #             }
    #             batch.update(post_transition_data, ts=t, bs=active_episodes)
    #
    #             # generate new observations
    #             output, o_ht_ct = self.run_obs_model(state.to(self.device), ht_ct=o_ht_ct)
    #             obs = output[:, 0, :obs_size].view(batch_size, 1, self.args.n_agents, self.agent_obs_size)
    #             avail_actions = output[:, 0, obs_size:].view(batch_size, 1, self.args.n_agents, self.args.n_actions)
    #
    #             # threshold avail_actions
    #             threshold = 0.5
    #             avail_actions = (avail_actions > threshold).float()
    #
    #             # handle cases where no agent actions are available e.g. when agent is dead
    #             mask = avail_actions.sum(-1) == 0
    #             source = torch.zeros_like(avail_actions)
    #             source[:, :, :, 0] = 1  # enable no-op
    #             avail_actions[mask] = source[mask]
    #
    #             # add pre-tranition data to the batch at the next timestep
    #             pre_transition_data = {
    #                 "state": batch_state[active_episodes],
    #                 "avail_actions": avail_actions[active_episodes],
    #                 "obs": obs[active_episodes]
    #             }
    #             batch.update(pre_transition_data, bs=active_episodes, ts=t+1)
    #
    #             # update active episodes
    #             active_episodes = [i for i, finished in enumerate(terminated.flatten()) if not finished]
    #             if all(terminated):
    #                 break
    #
    #             #print("\n=================================================\n")
    #
    #         #self.model_episodes += self.args.model_rollout_batch_size
    #
    #         return batch

    def cuda(self):
        if self.dynamics_model:
            self.dynamics_model.cuda()
        if self.representation_model:
            self.representation_model.cuda()
        if self.actions_model:
            self.actions_model.cuda()
        if self.policy_model:
            self.policy_model.cuda()

    def log_stats(self, t_env):
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("model_train_loss", self.train_loss, t_env)
            self.logger.log_stat("model_val_loss", self.val_loss, t_env)

            self.logger.log_stat("model_reward_train_loss", self.train_r_loss, t_env)
            self.logger.log_stat("model_term_train_loss", self.train_T_loss, t_env)
            self.logger.log_stat("model_available_actions_train_loss", self.train_aa_loss, t_env)
            self.logger.log_stat("model_policy_train_loss", self.train_p_loss, t_env)

            self.logger.log_stat("model_reward_val_loss", self.val_r_loss, t_env)
            self.logger.log_stat("model_term_val_loss", self.val_T_loss, t_env)
            self.logger.log_stat("model_available_actions_val_loss", self.val_aa_loss, t_env)
            self.logger.log_stat("model_policy_val_loss", self.val_p_loss, t_env)

            self.logger.log_stat("model_epsilon", self.mac.action_selector.epsilon, t_env)
            self.log_stats_t = t_env