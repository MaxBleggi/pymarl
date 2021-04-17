#TODO
#  - add l2 loss
#  - gradient tricks (see appendix G)
#  - prioritised replay
#  - can probably skip avail_actions model and rely on policy outputs to make invalid actions unlikely
#  - policy should use KL loss

import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from modules.models.dynamics import DynamicsModel
from modules.models.policy_actions import PolicyActionsModel
from modules.models.policy_actions import PADynamicsModel

import copy
import numpy as np
import random
from envs import REGISTRY as env_REGISTRY
import os
import pickle
from glob import glob
from controllers import REGISTRY as mac_REGISTRY
from .muzero_tree import *

class ModelMuZeroLearner:
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
        self.action_space = (self.args.n_agents, self.args.n_actions)
        self.state_size = args.state_shape - self.actions_size if args.env_args["state_last_action"] else args.state_shape
        self.reward_size = scheme["reward"]["vshape"][0]
        self.term_size = scheme["terminated"]["vshape"][0]
        self.value_size = 1

        # dynamics model
        self.dynamics_model = DynamicsModel(self.state_size, self.actions_size, args.dynamics_model_hidden_dim)

        # policy / actions model
        self.policy_actions_model = PolicyActionsModel(self.state_size, self.actions_size)
        self.pa_dynamics_model = PADynamicsModel(self.state_size, self.actions_size, args.policy_actions_model_hidden_dim)

        # optimizer
        self.dynamics_model_params = self.dynamics_model.parameters()
        self.policy_actions_model_params = list(self.policy_actions_model.parameters()) + list(self.pa_dynamics_model.parameters())
        
        self.dynamics_model_optimizer = torch.optim.Adam(self.dynamics_model_params, lr=self.args.model_learning_rate)
        self.policy_actions_model_optimizer = torch.optim.Adam(self.policy_actions_model_params, lr=self.args.model_learning_rate)

        # create copies fo the models to use for generating training data
        self._update_target_models()

        # logging stats
        self.train_loss, self.val_loss = 0, 0
        self.train_r_loss, self.val_r_loss = 0, 0
        self.train_T_loss, self.val_T_loss = 0, 0
        self.train_p_loss, self.val_p_loss = 0, 0
        self.train_aa_loss, self.val_aa_loss = 0, 0
        self.train_v_loss, self.val_v_loss = 0, 0
        self.model_grad_norm = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.epochs = 0
        self.save_index = 0
        self.training_steps = 0

        self.tree_stats = TreeStats()
        self.hidden_state = None # recurrent hidden state used throughout an episode
        self.trained = False

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

    def _update_target_models(self):
        self.target_dynamics_model = copy.deepcopy(self.dynamics_model)
        self.target_policy_actions_model = copy.deepcopy(self.policy_actions_model)
        self.target_pa_dynamics_model = copy.deepcopy(self.pa_dynamics_model)
        print("Update mcts target models")

    def get_episode_vars(self, batch):

        if batch.device != self.device:
            batch.to(self.device)

        # per-agent quantities
        obs = batch["obs"][:, :-1, ...]  # observations
        aa = batch["avail_actions"][:, :-1, ...].float()  # available actions
        action = batch["actions_onehot"][:, :-1, ...]  # actions taken
        mcts_policy = batch["mcts_policy"][:, :-1, ...]  # mcts visit counts

        # flatten per-agent quantities
        nbatch, ntimesteps, _, _ = obs.size()
        obs = obs.view((nbatch, ntimesteps, -1))
        aa = aa.view((nbatch, ntimesteps, -1))
        action = action.view((nbatch, ntimesteps, -1))

        # state
        state = batch["state"][:, :-1, :]
        if self.args.env_args["state_last_action"]:
            state = state[:, :, :self.state_size]

        # reward
        reward = batch["reward"][:, :-1, :]
        
        # value ie. n-step return
        value = torch.zeros_like(reward)
        n = self.args.model_bootstrap_timesteps
        coeff = torch.pow(self.args.gamma, torch.arange(0, n).float()).expand(nbatch, n).unsqueeze(-1).to(value.device)
        for i in range(0, ntimesteps):                
            r = reward[:, i : i + n]
            c = coeff[:, :r.size()[1]]        
            value[:, i] = torch.mul(r, c).sum(dim=1)

        # termination signal
        terminated = batch["terminated"][:, :-1].float()
        term_idx = terminated.max(1)[1] # terminal timestep
        term_signal = torch.ones_like(terminated) # mask timesteps including and after termination
        mask = torch.zeros_like(terminated) # active timesteps (includes termination timestep)
        for i in range(nbatch):
            term_signal[i, :term_idx[i]] = 0
            mask[i, :term_idx[i] + 1] = 1

        # generate target policy outputs
        policy = torch.zeros_like(action)

        if self.args.model_predict_target_policy:
            with torch.no_grad():
                self.target_mac.init_hidden(nbatch)
                for t in range(terminated.size()[1]): # max timesteps
                    policy[:, t, :] = self.target_mac.forward(batch, t=t).view(nbatch, -1)
        else:
            # otherwise approximate behaviour policy
            policy = torch.softmax(mcts_policy, dim=-1).view((nbatch, ntimesteps, -1))

        obs *= mask
        aa *= mask
        action *= mask
        reward *= mask
        state *= mask
        policy *= mask
        value *= mask

        return state, action, reward, term_signal, obs, aa, policy, value, mask

    def get_dynamics_model_input_output(self, state, actions, reward, term_signal, obs, aa, policy, value, mask,
                               state_prior_steps, start_t=0, predict=1, debug=False):

        # inputs
        s = state[:, :-1, :]  # state at time t
        a = actions[:, :-1, :]  # joint action at time t

        # outputs
        r = reward[:, :-1, :]  # reward at time t+1
        T = term_signal[:, :-1, :]  # terminated at t+1
        v = value[:, 1:, :]  # discounted k-step return at t+1

        y = torch.cat((r, v, T), dim=-1)

        start_s = max(0, start_t - state_prior_steps)
        end_s = start_t + predict
        s = s[:, start_s:end_s]

        a = a[:, start_s:end_s]
        y = y[:, start_t:start_t + predict]

        if debug:
            print(
                f"start_t={start_t}, prior_steps={state_prior_steps}, predict={predict}, s_slice={start_s}-{end_s}, s_size={s.size()}, a_slice={start_s}-{end_s}, a_size={a.size()}")

        return s, a, y

    def get_policy_actions_model_input_output(self, state, actions, reward, term_signal, obs, aa, policy, value, mask,
                                        state_prior_steps, start_t=0, predict=1, debug=False):

        # inputs
        s = state[:, :-1, :]  # state at time t
        a = actions[:, :-1, :]  # joint action at time t

        av = aa[:, :-1, :]  # available actions at t
        pt = policy[:, :-1, :]  # raw policy outputs at t

        y = torch.cat((pt, av), dim=-1)

        start_s = max(0, start_t - state_prior_steps)
        end_s = start_t + predict
        s = s[:, start_s:end_s]

        a = a[:, start_s:end_s]
        y = y[:, start_t:start_t + predict]

        if debug:
            print(
                f"start_t={start_t}, prior_steps={state_prior_steps}, predict={predict}, s_slice={start_s}-{end_s}, s_size={s.size()}, a_slice={start_s}-{end_s}, a_size={a.size()}")

        return s, a, y

    def run_dynamics_model(self, state, actions, ht_ct=None, predict=1):

        bs, steps, _ = actions.size()
        warmup = max(0, steps - predict)
        yp = torch.zeros(bs, predict, 3).to(self.device)

        # init and warmup
        if not ht_ct:
            ht_ct = self.dynamics_model.init_hidden(bs, self.device)

        for t in range(0, warmup):
            at = actions[:, t, :]
            st = state[:, t, :]
            st, _, _, _, ht_ct = self.dynamics_model(st, at, ht_ct)

        # predict
        for t in range(warmup, steps):
            at = actions[:, t, :]
            st = state[:, t, :] if warmup == 0 else st
            st, rt, vt, tt, ht_ct = self.dynamics_model(st, at, ht_ct)

            yp[:, t - warmup, :] = torch.cat((rt, vt, tt), dim=-1)

        return yp, ht_ct

    def run_policy_actions_model(self, state, actions, ht_ct=None, predict=1):

        bs, steps, _ = actions.size()
        warmup = max(0, steps - predict)
        # print(actions.size(), warmup, predict)
        yp = torch.zeros(bs, predict, 2 * self.actions_size).to(self.device)

        # init and warmup
        ht_ct = self.pa_dynamics_model.init_hidden(bs, self.device)
        for t in range(0, warmup):
            at = actions[:, t, :]
            st = state[:, t, :]
            st, ht_ct = self.pa_dynamics_model(st, at, ht_ct)

        # predict
        for t in range(warmup, steps):
            at = actions[:, t, :]
            st = state[:, t, :] if warmup == 0 else st

            pt, aa = self.policy_actions_model(st)
            yp[:, t - warmup, :] = torch.cat((pt, aa), dim=-1)

            st, ht_ct = self.pa_dynamics_model(st, at, ht_ct)

        return yp, ht_ct

    def _validate(self, vars, max_t):
        t_start = time.time()

        self.dynamics_model.eval()
        self.policy_actions_model.eval()
        self.pa_dynamics_model.eval()

        predict = self.args.model_rollout_timesteps

        nt = max_t.item() - predict - 2
        timesteps = np.arange(nt)

        total_val_loss = np.zeros(nt)
        total_r_val_loss = np.zeros(nt)
        total_term_val_loss = np.zeros(nt)
        total_aa_val_loss = np.zeros(nt)
        total_policy_val_loss = np.zeros(nt)
        total_value_val_loss = np.zeros(nt)

        with torch.no_grad():

            for i in timesteps:

                # dynamics
                state, actions, y = self.get_dynamics_model_input_output(*vars, self.args.model_state_prior_steps, start_t=i, predict=predict)
                yp, _ = self.run_dynamics_model(state, actions, predict=predict)
                loss_vector = F.mse_loss(yp, y, reduction='none')

                idx = 0
                total_val_loss[i] = loss_vector.mean().cpu().numpy().item()
                total_r_val_loss[i] = loss_vector[:, :, idx:idx + self.reward_size].mean().item(); idx += self.reward_size
                total_value_val_loss[i] = loss_vector[:, :, idx:idx + self.value_size].mean().item(); idx += self.value_size
                total_term_val_loss[i] = loss_vector[:, :, idx:idx + self.term_size].mean().item(); idx += self.term_size

                # policy actions
                state, actions, y  = self.get_policy_actions_model_input_output(*vars, self.args.model_state_prior_steps, start_t=i, predict=predict)
                yp, _ = self.run_policy_actions_model(state, actions, predict=predict)
                loss_vector = F.mse_loss(yp, y, reduction='none')

                idx = 0
                total_val_loss[i] += loss_vector.mean().item()
                total_policy_val_loss[i] = loss_vector[:, :, idx:idx + self.actions_size].mean().item(); idx += self.actions_size
                total_aa_val_loss[i] = loss_vector[:, :, idx:idx + self.actions_size].mean().item(); idx += self.actions_size

            self.val_loss = total_val_loss.mean()
            self.val_r_loss = total_r_val_loss.mean()
            self.val_T_loss = total_term_val_loss.mean()
            self.val_p_loss = total_policy_val_loss.mean()
            self.val_aa_loss = total_aa_val_loss.mean()
            self.val_v_loss = total_value_val_loss.mean()

            # TODO
            # if self.args.model_save_val_data:
            #     # save input_outputs
            #     if os.path.exists(self.args.episode_dir):
            #         fname = os.path.join(self.args.episode_dir, f"input_output_{self.save_index + 1:06}.pkl")
            #         with open(fname, 'wb') as f:
            #             save_data = {
            #                 "n_agents": self.args.n_agents,
            #                 "n_actions": self.args.n_actions,
            #                 "y": y.cpu(),
            #                 "yp": yp.cpu(),
            #                 "val_loss": self.val_loss
            #             }
            #             pickle.dump(save_data, f)
            #         self.save_index += 1

        # report losses
        t_step = (time.time() - t_start)
        print("Environment Model:")
        print(f"train loss: {self.train_loss:.5f}, val loss: {self.val_loss:.5f} time: {t_step:.2f} s")
        print(f" -- reward: train {self.train_r_loss:.5f} val {self.val_r_loss:.5f}")
        print(f" -- term: train {self.train_T_loss:.5f} val {self.val_T_loss:.5f}")
        print(f" -- avail_actions: train {self.train_aa_loss:.5f} val {self.val_aa_loss:.5f}")
        print(f" -- policy: train {self.train_p_loss:.5f} val {self.val_p_loss:.5f}")
        print(f" -- value: train {self.train_v_loss:.5f} val {self.val_v_loss:.5f}")

    def _train(self, vars, max_t):
        # learning a termination signal is easier with unmasked input

        self.dynamics_model.train()
        self.policy_actions_model.train()
        self.pa_dynamics_model.train()

        predict = self.args.model_rollout_timesteps
        n = self.args.model_rollout_timestep_samples
        nt = max_t.item() - predict - 2
        timesteps = [0] + np.random.choice(np.arange(1, nt), n).tolist()

        total_train_loss = np.zeros(n + 1)
        total_r_train_loss = np.zeros(n + 1)
        total_term_train_loss = np.zeros(n + 1)
        total_aa_train_loss = np.zeros(n + 1)
        total_policy_train_loss = np.zeros(n + 1)
        total_value_train_loss = np.zeros(n + 1)

        for i, t in enumerate(timesteps):

            # dynamics
            state, actions, y = self.get_dynamics_model_input_output(*vars, self.args.model_state_prior_steps, start_t=t, predict=predict)
            yp, _ = self.run_dynamics_model(state, actions, predict=predict)
            loss_vector = F.mse_loss(yp, y, reduction='none')

            idx = 0
            total_train_loss[i] = loss_vector.mean().item()
            total_r_train_loss[i] = loss_vector[:, :, idx:idx + self.reward_size].mean().item(); idx += self.reward_size
            total_value_train_loss[i] = loss_vector[:, :, idx:idx + self.value_size].mean().item(); idx += self.value_size
            total_term_train_loss[i] = loss_vector[:, :, idx:idx + self.term_size].mean().item(); idx += self.term_size

            # update dynamics model params
            self.dynamics_model_optimizer.zero_grad()
            loss_vector.mean().backward()
            self.dynamics_model_grad_norm = torch.nn.utils.clip_grad_norm_(self.dynamics_model_params, self.args.model_grad_clip_norm)
            self.dynamics_model_optimizer.step()

            # policy actions
            state, actions, y = self.get_policy_actions_model_input_output(*vars, self.args.model_state_prior_steps, start_t=t, predict=predict)
            yp, _ = self.run_policy_actions_model(state, actions, predict=predict)
            loss_vector = F.mse_loss(yp, y, reduction='none')

            idx = 0
            total_train_loss[i] += loss_vector.mean().item()
            total_policy_train_loss[i] = loss_vector[:, :, idx:idx + self.actions_size].mean().item(); idx += self.actions_size
            total_aa_train_loss[i] = loss_vector[:, :, idx:idx + self.actions_size].mean().item(); idx += self.actions_size

            # update policy action model params
            self.policy_actions_model_optimizer.zero_grad()
            loss_vector.mean().backward()
            self.policy_actions_model_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_actions_model_params, self.args.model_grad_clip_norm)
            self.policy_actions_model_optimizer.step()

        self.train_loss = total_train_loss.mean()
        self.train_r_loss = total_r_train_loss.mean()
        self.train_T_loss = total_term_train_loss.mean()
        self.train_aa_loss = total_aa_train_loss.mean()
        self.train_p_loss = total_policy_train_loss.mean()
        self.train_v_loss = total_value_train_loss.mean()

    def train(self, buffer):

        # train models
        batch = buffer.sample(min(buffer.episodes_in_buffer, self.args.model_batch_size))

        # Truncate batch to only filled timestep
        max_ep_t = batch.max_t_filled()
        batch = batch[:, :max_ep_t]

        if batch.device != self.args.device:
            batch.to(self.args.device)

        vars = self.get_episode_vars(batch)
        self._train(vars, max_ep_t)
        self.training_steps += 1

        if not self.trained:
            self.trained = True

        self.epochs += 1

        # validate periodically
        if (self.epochs + 1) % self.args.model_log_epochs == 0:
            # validate models
            batch = buffer.sample(min(buffer.episodes_in_buffer, self.args.model_batch_size))

            if batch.device != self.args.device:
                batch.to(self.args.device)

            max_ep_t = batch.max_t_filled()
            batch = batch[:, :max_ep_t]

            vars = self.get_episode_vars(batch)
            self._validate(vars, max_ep_t)

            self.epochs = 0

        # update targets
        # update target models if needed
        if self.training_steps == self.args.mcts_update_interval:
            self._update_target_models()
            self.training_steps = 0

    def select_child(self, parent, t_env):
        """
        use tree policy to select a child
        """
        action = None
        if self.args.model_mcts_tree_policy == "epsilon_greedy":
            action = self.select_epsilon_greedy_action(parent, t_env)
        elif self.args.model_mcts_tree_policy == "ucb":
            action = self.select_ucb_action(parent)

        key = tuple(action.flatten().cpu().tolist())
        #print('selecting greedy action', key)
        if key in parent.children:
            return parent.children[key]
        else:
            # if len(parent.children) > 0:
            #     print(f"new action selected from {parent.name}")
            child = Node(key, self.action_space, action=action, t=parent.t+1, device=self.device)
            parent.add_child(key, child)
            return child

    def select_ucb_action(self, parent):
        """
        UCB action selection
        """
        device = self.device
        c1 = self.args.ucb_c1
        c2 = self.args.ucb_c2
        batch_size = self.args.model_rollout_batch_size

        parent_count = torch.ones(self.action_space, device=device) * parent.count
        visit_ratio = torch.sqrt(parent_count) / (1 + parent.child_visits)
        c2_ratio = (parent_count + c2 + 1) / c2
        exploration_bonus = visit_ratio * (c1 + torch.log(c2_ratio)) if parent.count > 0 else 1.0
        prior_scores = torch.softmax(parent.priors, dim=-1) * exploration_bonus

        value_scores = parent.child_rewards + self.args.gamma * self.tree_stats.normalize(parent.action_values)
        scores = prior_scores + value_scores
        scores = scores.repeat(batch_size, 1, 1)

        # scores[parent.state.avail_actions == 0.0] = -float("inf")
        # adjust according to likelihood of actions being available
        scores = scores * parent.state.avail_actions
        selected_actions = torch.argmax(scores, dim=-1)

        # print(f"parent={parent.name}")
        # print(f"priors=\n{parent.priors.view(self.action_space)}")
        # print(f"prior_score=\n{prior_scores.view(self.action_space)}")
        # print(f"value_score=\n{value_scores.view(self.action_space)}")
        # print(f"avail_actions=\n{parent.state.avail_actions.view(self.action_space)}")
        # print(f"score=\n{scores.view(self.action_space)}")
        # print(f"selected_actions={selected_actions.flatten().cpu().tolist()}")
        # print("")

        return selected_actions

    def select_epsilon_greedy_action(self, parent, t_env):

        "epsilion greed with respect to parent priors"

        batch_size = self.args.model_rollout_batch_size
        avail_actions = parent.state.avail_actions

        if self.args.model_mcts_tree_policy_metric == "priors":
            return self.model_mac.select_actions(parent.priors.repeat(batch_size, 1, 1), avail_actions, t_env=t_env)
        elif self.args.model_mcts_tree_policy_metric == "counts":
            return self.model_mac.select_actions(parent.child_visits.repeat(batch_size, 1, 1), avail_actions, t_env=t_env)
        elif self.args.model_mcts_tree_policy_metric == "values":
            values = parent.child_rewards + self.args.gamma * self.tree_stats.normalize(parent.action_values)
            return self.model_mac.select_actions(values.repeat(batch_size, 1, 1), avail_actions, t_env=t_env)

    def select_action(self, parent, t_env=0, greedy=False):

        "root node action selection"

        device = self.device
        batch_size = self.args.model_rollout_batch_size
        avail_actions = parent.state.avail_actions
        actions, policy = None, None

        if self.args.model_mcts_root_policy_metric == "priors": # this only works with model_predict_target_policy=True
            actions = self.model_mac.select_actions(parent.priors.repeat(batch_size, 1, 1), avail_actions, t_env=t_env, greedy=greedy)
            policy = torch.zeros_like(parent.priors)
            if not self.args.model_predict_target_policy:
                print("Warning, model_mcts_root_policy_metric=priors should only be used with model_predict_target_policy=True")
        elif self.args.model_mcts_root_policy_metric == "counts":
            policy = torch.softmax(parent.child_visits, dim=-1)
            actions = self.model_mac.select_actions(policy.repeat(batch_size, 1, 1), avail_actions, t_env=t_env, greedy=greedy)
        elif self.args.model_mcts_root_policy_metric == "values":
            policy = parent.child_rewards + self.args.gamma * self.tree_stats.normalize(parent.action_values)
            actions = self.model_mac.select_actions(policy.repeat(batch_size, 1, 1), avail_actions, t_env=t_env, greedy=greedy)

        return actions, policy

    # def initialise_episode(self):
    #     batch_size = self.args.model_rollout_batch_size
    #     if self.args.model_select_action_from_target_policy:
    #         self.target_mac.init_hidden(batch_size)
    #     else:
    #         pass
    #         # self.hidden_state = self.target_dynamics_model.init_hidden(batch_size, self.device)  # dynamics model hidden state

    def initialise_mcts(self, root, batch, t_start):
        # encode the real state
        if batch.device != self.device:
            batch.to(self.device)

        batch_size = 1 #self.args.model_rollout_batch_size
        n_agents, n_actions = self.action_space

        self.target_dynamics_model.eval()
        self.target_policy_actions_model.eval()

        with torch.no_grad():

            # get real starting states for the batch
            start_s = max(0, t_start - self.args.model_state_prior_steps)
            state = batch["state"][:, start_s:t_start+1, :self.state_size]
            actions = batch["actions_onehot"][:, start_s:t_start+1, :].view(1, t_start+1 - start_s, -1)
            avail_actions = batch["avail_actions"][0, t_start, :].view(1, n_agents, n_actions) # current available actions

            # # expand starting states into batch size
            # state = state.repeat(batch_size, 1, 1)
            # actions = actions.repeat(batch_size, 1, 1)

            # initialise and warmup models
            d_ht_ct = self.target_dynamics_model.init_hidden(batch_size, self.device)
            pa_ht_ct = self.target_pa_dynamics_model.init_hidden(batch_size, self.device)

            warmup = max(0, state.size()[1] - 1)
            for t in range(0, warmup):
                at = actions[:, t]
                d_st = state[:, t]
                pa_st = state[:, t]

                d_st, _, _, _, d_ht_ct = self.target_dynamics_model(d_st, at, d_ht_ct)
                pa_st, pa_ht_ct = self.target_pa_dynamics_model(pa_st, at, pa_ht_ct)


            # initialise root node policy
            if self.args.model_select_action_from_target_policy:
                initial_priors = self.target_mac.forward(batch, t=t_start).view(batch_size, n_agents, n_actions)
            else:
                pa_st = state[:, 0] if warmup == 0 else pa_st
                d_st = state[:, 0] if warmup == 0 else d_st
                initial_priors, _ = self.target_policy_actions_model(pa_st)
                initial_priors = initial_priors.view(batch_size, n_agents, n_actions)

            # if t_start == 0:
            #     print("priors=", initial_priors)

            if self.args.model_add_exploration_noise:
                initial_priors = self.add_exploration_noise(initial_priors, avail_actions)
            #initial_priors = initial_priors * avail_actions

        root.priors = initial_priors
        root.state = TreeState((d_st, d_ht_ct), (pa_st, pa_ht_ct), avail_actions)
        root.expanded = True

    def expand_node(self, node):
        action = node.action.repeat(self.args.model_rollout_batch_size, 1)
        policy, value, reward, terminated, state = self.rollout(node.parent, action)
        node.update(policy, value, reward, terminated, state)
        node.expanded = True

    def backup(self, history):

        G = history[0][1].value # leaf value
        for parent, child in history:
            # print(f"backing up G={G} + r={child.reward} from {child.name} to {parent.name}")
            G = child.reward + self.args.gamma * G
            parent.backup(child.name, child.reward, G)
            self.tree_stats.update(G)
        # print(f"Backup finished at {history[-1][0].name} with G={history[-1][0].debug_return}")

    def add_exploration_noise(self, values, mask):
        noise = Dirichlet(mask.clone() * self.args.model_dirichlet_alpha)
        return values + noise.sample()

    def mcts(self, batch, t_env, t_start):

        self.tree_stats.clear()
        n_sim = self.args.model_mcts_simulations
        #print(f"Performing {n_sim} MCTS iterations")

        # initialise root node
        root = Node('root', self.action_space, t=t_start, device=self.device)
        self.initialise_mcts(root, batch, t_start)

        # run mcts
        debug = False

        for i in range(n_sim):
            #print(f"Simulation {i + 1}")
            child = parent = root
            depth = 0
            history = []

            while parent.expanded:
                # execute tree policy
                child = self.select_child(parent, t_env)
                if debug:
                    print(f"depth={depth}, parent={parent.name}, action={child.action.flatten().tolist()}, term={parent.terminated}")
                history.append((parent, child))
                parent = child
                depth += 1

            # expand current node
            if not child.parent.terminated:
                self.expand_node(child)

            history.reverse()
            self.backup(history)
            #print(f"expected={expected_return}, backup={root.debug_return}")
            if debug:
                print("----------------------------------------------")

        # select greedy action
        actions, policy = self.select_action(root, t_env=t_env, greedy=False)
        if t_start == 0:
            # root.summary()
            # print("aa=", root.state.avail_actions)
            print("priors=", root.priors)
            print("visits=", root.child_visits)
            # print("rewards=", root.child_rewards)
            print("values=", root.action_values)
            policy = root.child_rewards + self.args.gamma * self.tree_stats.normalize(root.action_values)
            print("policy=", policy)
            print("selected=", actions.flatten().tolist())

        return actions, policy

    def rollout(self, node, actions):

        d_state, pa_state, _ = node.state.totuple()
        d_st, d_ht_ct = d_state
        pa_st, pa_ht_ct = pa_state

        batch_size = self.args.model_rollout_batch_size
        n_agents, n_actions = self.action_space

        self.target_dynamics_model.eval()
        self.target_policy_actions_model.eval()
        self.target_pa_dynamics_model.eval()

        with torch.no_grad():

            # generate next state, reward, termination signal
            actions_onehot = F.one_hot(actions, num_classes=n_actions)

            # step dynamics model for reward, value and term
            d_st, reward, value, term_signal, d_ht_ct = self.target_dynamics_model(d_st, actions_onehot.view(batch_size, -1).float(), d_ht_ct)

            # get policy and avail actions
            policy, avail_actions = self.target_policy_actions_model(pa_st)
            policy = policy.view(batch_size, n_agents, n_actions)
            avail_actions = avail_actions.view(batch_size, n_agents, n_actions)
            # avail_actions = (avail_actions > self.args.model_action_threshold).int()

            # step policy dynamics
            pa_st = self.target_pa_dynamics_model(pa_ht_ct)

            # # clamp reward
            reward[reward < 0] = 0

            # generate termination mask
            terminated = (term_signal > self.args.model_term_threshold).item()

            # if this is the last allowable timestep, terminate
            if node.t == self.args.episode_limit - 1:
                terminated = True


            # handle cases where no agent actions are available e.g. when agent is dead
            mask = avail_actions.sum(-1) == 0
            source = torch.zeros_like(avail_actions)
            source[:, :, 0] = 1  # enable no-op
            avail_actions[mask] = source[mask]

            tree_state = TreeState((d_st, d_ht_ct), (pa_st, pa_ht_ct), avail_actions)
            return policy, value, reward, terminated, tree_state

    def cuda(self):
        if self.dynamics_model:
            self.dynamics_model.cuda()
        if self.policy_actions_model:
            self.policy_actions_model.cuda()
        if self.pa_dynamics_model:
            self.pa_dynamics_model.cuda()

        self._update_target_models()

    def log_stats(self, t_env):
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("model_train_loss", self.train_loss, t_env)
            self.logger.log_stat("model_val_loss", self.val_loss, t_env)
            self.logger.log_stat("dynamics_model_grad_norm", self.dynamics_model_grad_norm.cpu().numpy().item(), t_env)
            self.logger.log_stat("policy_actions_model_grad_norm", self.policy_actions_model_grad_norm.cpu().numpy().item(), t_env)

            self.logger.log_stat("model_reward_train_loss", self.train_r_loss, t_env)
            self.logger.log_stat("model_term_train_loss", self.train_T_loss, t_env)
            self.logger.log_stat("model_available_actions_train_loss", self.train_aa_loss, t_env)
            self.logger.log_stat("model_policy_train_loss", self.train_p_loss, t_env)
            self.logger.log_stat("model_value_train_loss", self.train_v_loss, t_env)

            self.logger.log_stat("model_reward_val_loss", self.val_r_loss, t_env)
            self.logger.log_stat("model_term_val_loss", self.val_T_loss, t_env)
            self.logger.log_stat("model_available_actions_val_loss", self.val_aa_loss, t_env)
            self.logger.log_stat("model_policy_val_loss", self.val_p_loss, t_env)
            self.logger.log_stat("model_value_val_loss", self.val_v_loss, t_env)

            self.logger.log_stat("model_epsilon", self.model_mac.action_selector.epsilon, t_env)
            self.log_stats_t = t_env