import time
import torch
import torch.nn.functional as F

from modules.models.representation import RepresentationModel
from modules.models.dynamics import DynamicsModel
from modules.models.actions import ActionsModel
from modules.models.policy import PolicyModel

import numpy as np
import random
from envs import REGISTRY as env_REGISTRY
import os
import pickle
from glob import glob
from controllers import REGISTRY as mac_REGISTRY


class ModelLearner:
    def __init__(self, mac, scheme, groups, logger, args):

        self.target_mac = mac # this is the actual policy network which we learn to emulate
        self.model_mac = mac_REGISTRY[args.model_mac](args)
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
        self.train_return_loss, self.val_return_loss = 0, 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.epochs = 0
        self.save_index = 0

        # debugging
        if self.args.model_save_val_data:
            # setup dir and clear existing files
            if self.args.episode_dir:
                if os.path.exists(self.args.episode_dir):
                    files = glob(os.path.join(self.args.episode_dir, "input_output_*.pkl"))
                    for f in files:
                        os.remove(f)
                else:
                    os.mkdir(self.args.episode_dir)
            else:
                raise Exception("Please specify 'episode_dir' or set 'save_episodes' to False")

    def train_test_split(self, indices, test_ratio=0.1, shuffle=True):

        if shuffle:
            random.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(test_ratio * n))
        train_indices = range(n - n_test)
        test_indices = range(len(train_indices), n)

        return train_indices, test_indices

    def get_episode_vars(self, batch):

        if batch.device != self.device:
            batch.to(self.device)

        # per-agent quantities
        obs = batch["obs"][:, :-1, ...]  # observations
        aa = batch["avail_actions"][:, :-1, ...].float()  # available actions
        action = batch["actions_onehot"][:, :-1, ...]  # actions taken

        # flatten per-agent quantities
        nbatch, ntimesteps, _, _ = obs.size()
        obs = obs.view((nbatch, ntimesteps, -1))
        aa = aa.view((nbatch, ntimesteps, -1))
        action = action.view(nbatch, ntimesteps, -1)

        # state
        state = batch["state"][:, :-1, :]
        if self.args.env_args["state_last_action"]:
            state = state[:, :, :self.state_size]

        # reward
        reward = batch["reward"][:, :-1, :]

        # termination signal
        terminated = batch["terminated"][:, :-1].float()
        term_idx = terminated.max(1)[1] # terminal timestep
        term_signal = torch.ones_like(terminated) # mask timesteps including and after termination
        mask = torch.zeros_like(terminated) # active timesteps (includes termination timestep)
        for i in range(nbatch):
            term_signal[i, :term_idx[i]] = 0
            mask[i, :term_idx[i] + 1] = 1

        # generate current policy outputs
        policy = torch.zeros_like(action)
        with torch.no_grad():
            self.target_mac.init_hidden(nbatch)
            for t in range(terminated.size()[1]): # max timesteps
                policy[:, t, :] = self.target_mac.forward(batch, t=t).view(nbatch, -1)

        obs *= mask
        aa *= mask
        action *= mask
        reward *= mask
        state *= mask
        policy *= mask

        return state, action, reward, term_signal, obs, aa, policy, mask

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

    def _validate(self, vars):
        t_start = time.time()

        self.representation_model.eval()
        self.dynamics_model.eval()
        self.actions_model.eval()
        self.policy_model.eval()

        with torch.no_grad():
            state, actions, y = self.get_model_input_output(*vars)
            yp, _ = self.run_model(state, actions)
            loss_vector = F.mse_loss(yp, y, reduction='none')

            self.val_loss = loss_vector.mean()
            self.val_r_loss = loss_vector[:, :, 0].mean()
            self.val_T_loss = loss_vector[:, :, 1].mean()
            self.val_aa_loss = loss_vector[:, :, 2:self.actions_size].mean()
            self.val_p_loss = loss_vector[:, :, 2 + self.actions_size:].mean()
            self.val_return_loss = F.mse_loss(yp[:, :, 0].sum(dim=1), y[:, :, 0].sum(dim=1))

            if self.args.model_save_val_data:
                # save input_outputs
                if os.path.exists(self.args.episode_dir):
                    fname = os.path.join(self.args.episode_dir, f"input_output_{self.save_index + 1:06}.pkl")
                    with open(fname, 'wb') as f:
                        save_data = {
                            "n_agents": self.args.n_agents,
                            "n_actions": self.args.n_actions,
                            "y": y.cpu(),
                            "yp": yp.cpu(),
                            "val_loss": self.val_loss.item()
                        }
                        pickle.dump(save_data, f)
                    self.save_index += 1


        # report losses
        t_step = (time.time() - t_start)
        print("Environment Model:")
        print(f"train loss: {self.train_loss:.5f}, val loss: {self.val_loss:.5f} time: {t_step:.2f} s")
        print(f" -- reward: train {self.train_r_loss:.5f} val {self.val_r_loss:.5f}")
        print(f" -- term: train {self.train_T_loss:.5f} val {self.val_T_loss:.5f}")
        print(f" -- avail_actions: train {self.train_aa_loss:.5f} val {self.val_aa_loss:.5f}")
        print(f" -- policy: train {self.train_p_loss:.5f} val {self.val_p_loss:.5f}")
        print(f" -- return: train {self.train_return_loss:.5f} val {self.val_return_loss:.5f}")

    def _train(self, vars):
        # learning a termination signal is easier with unmasked input

        self.representation_model.train()
        self.dynamics_model.train()
        self.actions_model.train()
        self.policy_model.train()

        # get data
        state, actions, y = self.get_model_input_output(*vars)

        for e in range(self.args.model_epochs):

            # make predictions
            yp, _ = self.run_model(state, actions)

            # gradient descent
            self.optimizer.zero_grad()
            loss_vector = F.mse_loss(yp, y, reduction='none')
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
            self.train_return_loss = F.mse_loss(yp[:, :, 0].sum(dim=1), y[:, :, 0].sum(dim=1))

    def train(self, buffer):

        # train models
        batch = buffer.sample(self.args.model_batch_size)

        if batch.device != self.args.device:
            batch.to(self.args.device)

            vars = self.get_episode_vars(batch)
            self._train(vars)

        self.epochs += 1

        # validate periodically
        if (self.epochs + 1) % self.args.model_log_epochs == 0:
            # validate models
            batch = buffer.sample(self.args.model_batch_size)

            if batch.device != self.args.device:
                batch.to(self.args.device)

                vars = self.get_episode_vars(batch)
                self._validate(vars)
            self.epochs = 0

    def generate_batch(self, buffer, t_env):

        # sample starts from episode buffer
        if not buffer.can_sample(1):
            return

        # start with one episode as the seed since all episodes have the same starts
        episode = buffer.sample(1)
        if episode.device != self.device:
            episode.to(self.device)

        batch_size = self.args.model_rollout_batch_size

        self.representation_model.eval()
        self.dynamics_model.eval()
        self.actions_model.eval()
        self.policy_model.eval()

        with torch.no_grad():

            # get real starting states for the batch
            state = episode["state"][:, 0, :self.state_size]
            avail_actions = episode["avail_actions"][:, 0]
            _, n_agents, n_actions = avail_actions.size()
            term_signal = episode["terminated"][:, 0].float()

            # expand starting states into batch size
            state = state.repeat(batch_size, 1)
            avail_actions = avail_actions.repeat(batch_size, 1, 1)
            term_signal = term_signal.repeat(batch_size, 1)

            # track active episodes
            terminated = (term_signal > 0)
            active_episodes = [i for i, finished in enumerate(terminated.flatten()) if not finished]

            max_t = episode.max_seq_length - 1

            # initialise hidden states
            ht, ct = self.dynamics_model.init_hidden(batch_size, self.device) # dynamics model hidden state

            # reward distribution
            G = torch.zeros(batch_size, 1).to(self.device)

            # action history
            H = torch.zeros(batch_size, max_t, n_agents).to(self.device)

            for t in range(0, max_t):
                if t == 0:
                    ht = self.representation_model(state)

                # choose actions following current policy
                agent_outputs = self.policy_model(ht).view(batch_size, n_agents, n_actions)
                actions = self.model_mac.select_actions(agent_outputs, avail_actions, t_env)
                actions_onehot = F.one_hot(actions, num_classes=n_actions)

                # update action history
                H[:, t, ...] = actions

                # generate next state, reward, termination signal
                rT, (ht, ct) = self.dynamics_model(actions_onehot.view(batch_size, -1).float(), (ht, ct))
                reward = rT[:, 0]
                term_signal = rT[:, 1]

                # clamp reward
                reward[reward < 0] = 0

                # add reward to episode returns
                G[:, 0] += reward
                # generate termination mask
                threshold = self.args.model_term_threshold
                terminated = (term_signal > threshold)

                # if this is the last timestep, terminate
                if t == max_t - 1:
                    terminated[active_episodes] = True

                # generate and threshold avail_actions
                avail_actions = self.actions_model(ht).view(batch_size, n_agents, n_actions)
                threshold = self.args.model_action_threshold
                avail_actions = (avail_actions > threshold).int()

                # handle cases where no agent actions are available e.g. when agent is dead
                mask = avail_actions.sum(-1) == 0
                source = torch.zeros_like(avail_actions)
                source[:, :, 0] = 1  # enable no-op
                avail_actions[mask] = source[mask]

                # update active episodes
                active_episodes = [i for i, finished in enumerate(terminated.flatten()) if not finished]
                if all(terminated):
                    break

            print(f"Collected {self.args.model_rollout_batch_size} episodes from MODEL ENV using epsilon: "
                  f"{self.model_mac.action_selector.epsilon:.3f}")

            # # histogram plotting
            # f, bins = np.histogram(G.cpu(), bins=5)
            # total = sum(f)
            # for i, count in enumerate(f):
            #     start = bins[i]
            #     end = bins[i + 1]
            #     p = int(100 * count / total)
            #     if p == 0 and count > 0:
            #         p = 1
            #     bar = "".join(["+"] * p)
            #     print(f"{start:<5.1f}-{end:5.1f}  {count:<3} {bar}")
            # print()
            G_cpu = G.cpu()
            print(f"RETURNS: mean: {G_cpu.mean():.2f} std: {G_cpu.std():.2f} max: {G_cpu.max():.2f} min: {G_cpu.min():.2f}")
            print(f" -- losses: train {self.train_return_loss:.5f} val {self.val_return_loss:.5f}")
            return H, G

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

            self.logger.log_stat("model_epsilon", self.model_mac.action_selector.epsilon, t_env)
            self.log_stats_t = t_env