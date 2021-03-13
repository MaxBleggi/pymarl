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

from modules.models.representation import RepresentationModel
from modules.models.dynamics import DynamicsModel
from modules.models.actions import ActionsModel
from modules.models.policy import PolicyModel
from modules.models.reward import RewardModel
from modules.models.term import TermModel
from modules.models.value import ValueModel

import copy
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

        # create copies fo the models to use for generating training data
        self._update_target_models()

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
        self.training_steps = 0

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

    def _update_target_models(self):
        self.target_dynamics_model = copy.deepcopy(self.dynamics_model)
        self.target_representation_model = copy.deepcopy(self.representation_model)
        self.target_policy_model = copy.deepcopy(self.policy_model)
        self.target_reward_model = copy.deepcopy(self.reward_model)
        self.target_value_model = copy.deepcopy(self.value_model)
        self.target_actions_model = copy.deepcopy(self.actions_model)
        self.target_term_model = copy.deepcopy(self.term_model)
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
            value[:, i] = r.sum(dim=1)

        # termination signal
        terminated = batch["terminated"][:, :-1].float()
        term_idx = terminated.max(1)[1] # terminal timestep
        term_signal = torch.ones_like(terminated) # mask timesteps including and after termination
        mask = torch.zeros_like(terminated) # active timesteps (includes termination timestep)
        for i in range(nbatch):
            term_signal[i, :term_idx[i]] = 0
            mask[i, :term_idx[i] + 1] = 1

        # generate target policy outputs
        # use this to approximate target policy
        # with torch.no_grad():
        #     self.target_mac.init_hidden(nbatch)
        #     for t in range(terminated.size()[1]): # max timesteps
        #         policy[:, t, :] = self.target_mac.forward(batch, t=t).view(nbatch, -1)

        # otherwise approximate behaviour policy
        policy = torch.softmax(mcts_policy, dim=-1).view((nbatch, ntimesteps, -1))
        # policy = mcts_policy.view((nbatch, ntimesteps, -1))

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

        # outputs
        r = reward[:, :-1, :]  # reward at time t+1        
        T = term_signal[:, :-1, :]  # terminated at t+1

        vt = value[:, 1:, :]  # discounted k-step return at t+1
        av = aa[:, 1:, :]  # available actions at t+1
        pt = policy[:, 1:, :] # raw policy outputs at t+1

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

            # step
            ht, ct = self.dynamics_model(at, (ht, ct))

            rt = self.reward_model(ht)            
            Tt = self.term_model(ht)
            avt = self.actions_model(ht)
            pt = self.policy_model(ht)
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
            self._validate(vars)

            self.epochs = 0

        # update targets
        # update target models if needed
        if self.training_steps == self.args.mcts_update_interval:
            self._update_target_models()
            self.training_steps = 0

    def select_child(self, parent):
        """
        use tree policy to select a child via max ucb
        """
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
        prior_scores = parent.priors * exploration_bonus

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

    def select_action(self, parent, t_env=0, greedy=False):
        """
        Temperature based action selection based on visit count
        """
        device = self.device
        batch_size = self.args.model_rollout_batch_size

        counts = parent.child_visits.repeat(batch_size, 1, 1)

        avail_actions = parent.state.avail_actions
        selected_actions = self.model_mac.select_actions(counts, avail_actions, t_env=t_env, greedy=greedy)

        # print(f"name={parent.name}, counts={counts.view(self.action_space)}")
        # print(f"avail_actions={parent.state.avail_actions.view(self.action_space)}")
        # print(f"selected_action={selected_actions}")

        return selected_actions

    def initialise(self, batch, t_start):
        # encode the real state
        if batch.device != self.device:
            batch.to(self.device)

        batch_size = self.args.model_rollout_batch_size
        n_agents, n_actions = self.action_space

        self.target_representation_model.eval()
        self.target_dynamics_model.eval()
        self.target_policy_model.eval()

        with torch.no_grad():

            # get real starting states for the batch
            start_s = max(0, t_start - self.args.model_state_prior_steps)
            state = batch["state"][:, start_s:t_start+1, :self.state_size]
            avail_actions = batch["avail_actions"][:, t_start]

            # expand starting states into batch size
            state = state.repeat(batch_size, 1, 1)
            avail_actions = avail_actions.repeat(batch_size, 1, 1)

            # generate implicit state
            ht = self.target_representation_model(state)

            # initialise dynamics model hidden state
            _, ct = self.target_dynamics_model.init_hidden(batch_size, self.device)  # dynamics model hidden state

            # initialise root node policy
            initial_priors = self.target_policy_model(ht).view(batch_size, n_agents, n_actions)
            if t_start == 0:
                print("priors=", initial_priors)

            initial_priors = self.add_exploration_noise(initial_priors, avail_actions)
            initial_priors = initial_priors * avail_actions

        return initial_priors, TreeState(ht, ct, avail_actions)

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
        initial_priors, root.state = self.initialise(batch, t_start)
        # print(f"initial_priors={initial_priors}")
        root.priors = initial_priors
        root.expanded = True

        # run mcts
        debug = False
        for i in range(n_sim):
            #print(f"Simulation {i + 1}")
            child = parent = root
            depth = 0
            history = []

            while parent.expanded:
                # execute tree policy
                child = self.select_child(parent)
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
        action = self.select_action(root, t_env=t_env, greedy=False)
        values = root.child_rewards + self.args.gamma * root.action_values

        if t_start == 0:

            # root.summary()
            # print("aa=", root.state.avail_actions)
            # print("visits=", root.child_visits)
            # print("rewards=", root.child_rewards)
            # print("values=", root.action_values)
            # print("priors=", initial_priors)
            print("values=", values)
            print("selected=", action.flatten().tolist())

        return action, values

    def rollout(self, node, actions):

        ht, ct, avail_actions = node.state.totuple()

        batch_size = self.args.model_rollout_batch_size
        n_agents, n_actions = self.action_space

        self.target_dynamics_model.eval()
        self.target_actions_model.eval()
        self.target_policy_model.eval()
        self.target_reward_model.eval()
        self.target_term_model.eval()
        self.target_value_model.eval()

        with torch.no_grad():

            # generate next state, reward, termination signal
            actions_onehot = F.one_hot(actions, num_classes=n_actions)

            # step environment
            ht, ct = self.target_dynamics_model(actions_onehot.view(batch_size, -1).float(), (ht, ct))

            # post transition predictions
            reward = self.target_reward_model(ht)
            value = self.target_value_model(ht)
            term_signal = self.target_term_model(ht)
            policy = self.target_policy_model(ht).view(batch_size, n_agents, n_actions)


            # # clamp reward
            # reward[reward < 0] = 0

            # generate termination mask
            terminated = (term_signal > self.args.model_term_threshold).item()

            # if this is the last allowable timestep, terminate
            if node.t == self.args.episode_limit - 1:
                terminated = True

            # generate and threshold avail_actions
            avail_actions = self.target_actions_model(ht).view(batch_size, n_agents, n_actions)
            # avail_actions = (avail_actions > self.args.model_action_threshold).int()

            # handle cases where no agent actions are available e.g. when agent is dead
            mask = avail_actions.sum(-1) == 0
            source = torch.zeros_like(avail_actions)
            source[:, :, 0] = 1  # enable no-op
            avail_actions[mask] = source[mask]

            return policy, value, reward, terminated, TreeState(ht, ct, avail_actions)

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

        self._update_target_models()

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