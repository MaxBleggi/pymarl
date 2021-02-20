#TODO
#  - add l2 loss
#  - gradient tricks (see appendix G)
#  - prioritised replay
#  - can probably skip avail_actions model and rely on policy outputs to make invalid actions unlikely
#  - dirchlect alpha?

import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from modules.models.representation import RepresentationModel
from modules.models.dynamics import DynamicsModel
from modules.models.actions import ActionsModel
from modules.models.policy import PolicyModel
from modules.models.reward import RewardModel
from modules.models.term import TermModel
from modules.models.value import ValueModel

import numpy as np
import random
from envs import REGISTRY as env_REGISTRY
import os
import pickle
from glob import glob
import queue
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
        dynamics_model_input_size = self.actions_size
        #self.dynamics_model_output_size = self.reward_size + self.term_size
        self.dynamics_model = DynamicsModel(dynamics_model_input_size, args.dynamics_model_hidden_dim)

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

        # reward model
        reward_model_input_size = args.dynamics_model_hidden_dim
        reward_model_output_size = self.reward_size
        self.reward_model = RewardModel(reward_model_input_size, reward_model_output_size, args.reward_model_hidden_dim)

        # term model
        term_model_input_size = args.dynamics_model_hidden_dim
        term_model_output_size = self.term_size
        self.term_model = TermModel(term_model_input_size, term_model_output_size, args.term_model_hidden_dim)

        # value model
        value_model_input_size = args.value_model_hidden_dim
        value_model_output_size = self.value_size
        self.value_model = ValueModel(value_model_input_size, value_model_output_size, args.value_model_hidden_dim)

        # optimizer
        self.params = list(self.dynamics_model.parameters()) + list(self.representation_model.parameters()) \
                    + list(self.actions_model.parameters()) + list(self.policy_model.parameters()) \
                    + list(self.reward_model.parameters()) + list(self.term_model.parameters()) \
                    + list(self.value_model.parameters())
        
        self.optimizer = torch.optim.Adam(self.params, lr=self.args.model_learning_rate)

        # logging stats
        self.train_loss, self.val_loss = 0, 0
        self.train_r_loss, self.val_r_loss = 0, 0
        self.train_T_loss, self.val_T_loss = 0, 0
        self.train_aa_loss, self.val_aa_loss = 0, 0
        self.train_p_loss, self.val_p_loss = 0, 0
        self.train_v_loss, self.val_v_loss = 0, 0
        self.model_grad_norm = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.epochs = 0
        self.save_index = 0

        self.tree_stats = TreeStats()
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
        
        # value ie. n-step return
        value = torch.zeros_like(reward)
        n = self.args.model_bootstrap_timesteps
        coeff = torch.pow(self.args.gamma, torch.arange(0, n).float()).expand(nbatch, n).unsqueeze(-1).to(value.device)
        for i in range(0, ntimesteps):                
            r = reward[:, i + 1:i + n + 1]
            c = coeff[:, :r.size()[1]]        
            # value[:, i] = torch.mul(r, c).sum(dim=1)
            value[:, i] = r.sum(dim=1)

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
        value *= mask

        return state, action, reward, term_signal, obs, aa, policy, value, mask

    def get_model_input_output(self, state, actions, reward, term_signal, obs, aa, policy, value, mask, start_t=0, max_t=None):

        # inputs
        s = state[:, :-1, :]  # state at time t
        a = actions[:, :-1, :]  # joint action at time t
        av = aa[:, :-1, :]  # available actions at t
        pt = policy[:, :-1, :] # raw policy outputs at t        

        # outputs
        r = reward[:, :-1, :]  # reward at time t+1        
        T = term_signal[:, :-1, :]  # terminated at t+1
        vt = value[:, :-1, :]  # estimated discounted k-step return at t 

        y = torch.cat((r, T, av, pt, vt), dim=-1)

        if start_t == "random":
            start = int(random.random() * (s.size()[1] - 1))
        else:
            start = start_t

        start_s = max(0, start - self.args.model_state_prior_steps)
        end_s = start_s + self.args.model_state_prior_steps
        s = s[:, start_s:end_s]

        a = a[:, start:]
        y = y[:, start:]

        if max_t:
            a = a[:, :max_t]
            y = y[:, :max_t]

        return s, a, y

    def run_model(self, state, actions, ht_ct=None):

        bs, steps, n_actions = actions.size()
        ht, ct = ht_ct if ht_ct else self.dynamics_model.init_hidden(bs, self.device)
        output_size = self.reward_size + self.term_size + 2 * self.actions_size + self.value_size
        yp = torch.zeros(bs, steps, output_size).to(self.device)

        for t in range(0, steps):

            if t == 0:
                ht = self.representation_model(state)

            at = actions[:, t, :]
            avt = self.actions_model(ht)
            pt = self.policy_model(ht) # TODO this could probably use avt as additional input            

            # step forward
            ht, ct = self.dynamics_model(at, (ht, ct))
            rt = self.reward_model(ht)            
            Tt = self.term_model(ht)
            vt = self.value_model(ht)

            yp[:, t, :] = torch.cat((rt, Tt, avt, pt, vt), dim=-1)

        return yp, (ht, ct)

    def _validate(self, vars):
        t_start = time.time()

        self.representation_model.eval()
        self.dynamics_model.eval()
        self.actions_model.eval()
        self.policy_model.eval()
        self.reward_model.eval()
        self.term_model.eval()
        self.value_model.eval()

        with torch.no_grad():
            state, actions, y = self.get_model_input_output(*vars, start_t="random", max_t=self.args.model_rollout_timesteps)
            yp, _ = self.run_model(state, actions)
            loss_vector = F.mse_loss(yp, y, reduction='none')

            idx = 0
            self.val_loss = loss_vector.mean().cpu().numpy().item()
            self.val_r_loss = loss_vector[:, :, idx:idx + self.reward_size].mean().cpu().numpy().item(); idx += self.reward_size
            self.val_T_loss = loss_vector[:, :, idx:idx + self.term_size].mean().cpu().numpy().item(); idx += self.term_size
            self.val_aa_loss = loss_vector[:, :, idx:idx + self.actions_size].mean().cpu().numpy().item(); idx += self.actions_size
            self.val_p_loss = loss_vector[:, :, idx:idx + self.actions_size].mean().cpu().numpy().item(); idx += self.actions_size
            self.val_v_loss = loss_vector[:, :, idx:idx + self.value_size].mean().cpu().numpy().item(); idx += self.value_size

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
        print(f" -- value: train {self.train_v_loss:.5f} val {self.val_v_loss:.5f}")

    def _train(self, vars, max_t):
        # learning a termination signal is easier with unmasked input

        self.representation_model.train()
        self.dynamics_model.train()
        self.actions_model.train()
        self.policy_model.train()
        self.term_model.train()
        self.value_model.train()

        rollout_timesteps = self.args.model_rollout_timesteps if self.args.model_rollout_timesteps else 0

        timesteps = list(range(max_t.item() - rollout_timesteps - 2))
        random.shuffle(timesteps)

        n = self.args.model_rollout_timestep_samples
        nt = max_t.item() - rollout_timesteps - 2
        timesteps = [0] + np.random.choice(np.arange(1, nt), n).tolist()

        self.train_loss = np.zeros(n + 1)
        self.train_r_loss = np.zeros(n + 1)
        self.train_T_loss = np.zeros(n + 1)
        self.train_aa_loss = np.zeros(n + 1)
        self.train_p_loss = np.zeros(n + 1)
        self.train_v_loss = np.zeros(n + 1)

        for i, t in enumerate(timesteps):

            # get data
            state, actions, y = self.get_model_input_output(*vars, start_t=t, max_t=self.args.model_rollout_timesteps)
            #print(t, max_t.item(), rollout_timesteps, t + rollout_timesteps, state.size())

            # make predictions
            yp, _ = self.run_model(state, actions)

            # gradient descent
            self.optimizer.zero_grad()
            loss_vector = F.mse_loss(yp, y, reduction='none')
            loss = loss_vector.mean()
            loss.backward()
            self.model_grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.model_grad_clip_norm)
            self.optimizer.step()

            # record losses
            idx = 0
            self.train_loss[i] = loss.item()
            self.train_r_loss[i] = loss_vector[:, :, idx:idx + self.reward_size].mean(); idx += self.reward_size
            self.train_T_loss[i] = loss_vector[:, :, idx:idx + self.term_size].mean(); idx += self.term_size
            self.train_aa_loss[i] = loss_vector[:, :, idx:idx + self.actions_size].mean(); idx += self.actions_size
            self.train_p_loss[i] = loss_vector[:, :, idx:idx + self.actions_size].mean(); idx += self.actions_size
            self.train_v_loss[i] = loss_vector[:, :, idx:idx + self.value_size].mean(); idx += self.value_size

        self.train_loss = self.train_loss.mean()
        self.train_r_loss = self.train_r_loss.mean()
        self.train_T_loss = self.train_T_loss.mean()
        self.train_aa_loss = self.train_aa_loss.mean()
        self.train_p_loss = self.train_p_loss.mean()
        self.train_v_loss = self.train_v_loss.mean()

    def train(self, buffer):

        # train models
        batch = buffer.sample(self.args.model_batch_size)

        # Truncate batch to only filled timestep
        max_ep_t = batch.max_t_filled()
        batch = batch[:, :max_ep_t]

        if batch.device != self.args.device:
            batch.to(self.args.device)

        vars = self.get_episode_vars(batch)
        self._train(vars, max_ep_t)

        if not self.trained:
            self.trained = True

        self.epochs += 1

        # validate periodically
        if (self.epochs + 1) % self.args.model_log_epochs == 0:
            # validate models
            batch = buffer.sample(self.args.model_batch_size)

            if batch.device != self.args.device:
                batch.to(self.args.device)

            max_ep_t = batch.max_t_filled()
            batch = batch[:, :max_ep_t]

            vars = self.get_episode_vars(batch)
            self._validate(vars)

            self.epochs = 0

    def select_child(self, parent):
        """
        use tree policy to select a child via max ucb
        """
        action = self.select_action(parent, greedy=True)
        key = tuple(action.flatten().cpu().tolist())
        #print('selecting greedy action', key)
        if key in parent.children:
            return parent.children[key]
        else:
            return None

    def select_action(self, parent, t_env=0, greedy=False):
        """
        UCB action selection
        """
        device = self.device
        c1 = self.args.ucb_c1
        c2 = self.args.ucb_c2
        batch_size = self.args.model_rollout_batch_size

        parent_count = torch.ones(self.action_space, device=device) * parent.count
        visit_ratio = torch.sqrt(parent_count) / (1 + parent.child_visits)
        # print('visit ratio')
        # print(visit_ratio)

        c2_ratio = (parent_count + c2 + 1) / c2
        # print('c2 ratio')
        # print(c2_ratio)

        nq = self.tree_stats.normalize(parent.action_values)
        scores =  nq + parent.priors * visit_ratio * (c1 + torch.log(c2_ratio))
        scores = scores.repeat(batch_size, 1, 1)
        # print('scores')
        # print(scores)

        #selected_actions = torch.argmax(scores, dim=1)
        selected_actions = self.model_mac.select_actions(scores, parent.state.avail_actions, t_env=t_env, greedy=greedy)
        # print('selected actions')
        # print(selected_actions)

        return selected_actions

    def initialise(self, batch, t_start):
        # encode the real state
        if batch.device != self.device:
            batch.to(self.device)

        batch_size = self.args.model_rollout_batch_size

        self.representation_model.eval()
        self.dynamics_model.eval()

        with torch.no_grad():

            # get real starting states for the batch
            start_s = max(0, t_start - self.args.model_state_prior_steps)
            state = batch["state"][:, start_s:t_start+1, :self.state_size]
            avail_actions = batch["avail_actions"][:, t_start]
            term_signal = batch["terminated"][:, t_start].float()

            # expand starting states into batch size
            state = state.repeat(batch_size, 1, 1)
            avail_actions = avail_actions.repeat(batch_size, 1, 1)
            term_signal = term_signal.repeat(batch_size, 1)

            # generate implicit state
            ht = self.representation_model(state)

            # initialise dynamics model hidden state
            _, ct = self.dynamics_model.init_hidden(batch_size, self.device)  # dynamics model hidden state

        return TreeState(ht, ct, avail_actions, term_signal)

    def expand_node(self, parent, t_env):
        # select a joint action and add it to the set of children
        action = self.select_action(parent, t_env=t_env)
        key = tuple(action.flatten().cpu().tolist())
        action = action.repeat(self.args.model_rollout_batch_size, 1)
        # child = self.simulate(parent, action)

        # simulate and store results in new child node
        child = Node(key, self.action_space, t=parent.t+1, device=self.device)
        Q, V, R, terminal, state = self.rollout(parent, action)
        self.tree_stats.update(Q)
        child.update(Q, V, R, terminal, state)
        parent.add_child(key, child)

        return child

    def backup(self, history):
        i = len(history) - 1
        leaf, _ = history.pop(0)
        G = (self.args.gamma ** i) * (leaf.reward + leaf.value) # todo: do we double count rt ?

        i -= 1
        for node, action in history:
            G += (self.args.gamma ** i) * node.reward
            #print(f" -- {str(node)}, {action}, r={node.reward}, i={i}, G={G}")
            node.backup(action, G)
            i -= 1

    def add_exploration_noise(self, x):
        m = Dirichlet(torch.ones_like(x) * self.args.model_dirichlet_alpha)
        return x + m.sample()

    def mcts(self, batch, t_env, t_start):

        self.tree_stats.clear()
        n_sim = self.args.model_mcts_simulations
        #print(f"Performing {n_sim} MCTS iterations")

        # initialise root node
        root = Node('root', self.action_space, t=t_start, device=self.device)
        root.state = self.initialise(batch, t_start) # TODO these tensors might need to be held in CPU memory
        root.priors = self.add_exploration_noise(root.priors)

        # run mcts
        for i in range(n_sim):
            #print(f"Simulation {i + 1}")
            parent = child = root
            depth = 0
            history = []

            while child:
                # execute tree policy
                #print(f"depth: {depth}: {parent}")
                child = self.select_child(parent)
                if child:
                    history.append((parent, child.name))
                    parent = child

            # expand current node
            child = self.expand_node(parent, t_env)
            history.append((parent, child.name))
            history.append((child, None))
            #print(f"expanded {parent.name} -> {child.name}")

            #print('backup ...')
            history.reverse()
            self.backup(history)
            #print("")

        #print(self.tree_stats.normalize(root.action_values))
        action = self.select_action(root, greedy=True)
        return action

    def rollout(self, node, actions):

        # preform model based rollout from the current node

        # get the current state variables and transfer to the correct device if necessary
        # batch = node.batch
        # for b in batch:
        #     if b.device != self.device:
        #         b.to(self.device)
        ht, ct, avail_actions, term_signal = node.state.totuple()
        batch_size = self.args.model_rollout_batch_size
        n_agents, n_actions = self.action_space

        self.dynamics_model.eval()
        self.actions_model.eval()
        self.policy_model.eval()
        self.reward_model.eval()
        self.term_model.eval()
        self.value_model.eval()

        with torch.no_grad():

            # track active episodes
            terminated = (term_signal > 0)
            active_episodes = [i for i, finished in enumerate(terminated.flatten()) if not finished]

            # set the number of rollout timesteps
            max_t = self.args.episode_limit - node.t - 1
            # k = self.args.model_rollout_timesteps if self.args.model_rollout_timesteps else max_t
            k = 1

            R = torch.zeros(batch_size, k, self.reward_size).to(self.device) # reward history
            V = torch.zeros(batch_size, k, self.value_size).to(self.device) # n-step return estimate
            Q = torch.zeros(batch_size, k, n_agents, n_actions).to(self.device) # action values

            #print(f"Rolling out {k} timesteps from t={node.t}")
            for t in range(0, k):

                # choose actions following current policy
                agent_outputs = self.policy_model(ht).view(batch_size, n_agents, n_actions)

                #actions = self.model_mac.select_actions(agent_outputs, avail_actions)
                actions_onehot = F.one_hot(actions, num_classes=n_actions)

                # update action values and action history
                Q[:, t, ...] = agent_outputs

                # generate next state, reward, termination signal
                ht, ct = self.dynamics_model(actions_onehot.view(batch_size, -1).float(), (ht, ct))
                reward = self.reward_model(ht)
                term_signal = self.term_model(ht)

                # # clamp reward
                # reward[reward < 0] = 0

                # record timestep reward
                R[:, t, :] = reward

                # generate termination mask
                terminated = (term_signal > self.args.model_term_threshold)

                # mask previously terminated episodes to avoid reactivatation
                inactive_episodes = list(set(range(batch_size)) - set(active_episodes))
                terminated[inactive_episodes] = False

                # if this is the last allowable timestep, terminate
                if t == max_t - 1:
                    terminated[active_episodes] = True

                # generate and threshold avail_actions
                avail_actions = self.actions_model(ht).view(batch_size, n_agents, n_actions)
                # avail_actions = (avail_actions > self.args.model_action_threshold).int()

                # handle cases where no agent actions are available e.g. when agent is dead
                mask = avail_actions.sum(-1) == 0
                source = torch.zeros_like(avail_actions)
                source[:, :, 0] = 1  # enable no-op
                avail_actions[mask] = source[mask]

                # update active episodes
                active_episodes = [i for i, finished in enumerate(terminated[active_episodes].flatten()) if not finished]
                if all(terminated):
                    break

            return Q, V, R, terminated, TreeState(ht, ct, avail_actions, term_signal)

    def cuda(self):
        if self.dynamics_model:
            self.dynamics_model.cuda()
        if self.representation_model:
            self.representation_model.cuda()
        if self.actions_model:
            self.actions_model.cuda()
        if self.policy_model:
            self.policy_model.cuda()
        if self.reward_model:
            self.reward_model.cuda()
        if self.term_model:
            self.term_model.cuda()
        if self.value_model:
            self.value_model.cuda()

    def log_stats(self, t_env):
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("model_train_loss", self.train_loss, t_env)
            self.logger.log_stat("model_val_loss", self.val_loss, t_env)
            self.logger.log_stat("model_grad_norm", self.model_grad_norm.cpu().numpy().item(), t_env)

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