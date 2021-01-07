from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import time

class ModelMCTSEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.rollout_steps = []

    def setup(self, scheme, groups, preprocess, mac, model=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.model = model

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, use_search=False, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        partial_return = 0
        expected_partial_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)


        search_initialised = False
        if use_search:
            print(f"Generating {self.args.model_rollout_batch_size} rollouts of depth {self.args.model_rollout_timesteps} with starting epsilon {self.model.model_mac.action_selector.epsilon:.3f}")
            t_op_start = time.time()

        #max_rerolls = max(1, int(np.sqrt(self.avg_rollouts)))
        rerolls = 0
        rollout_steps = 0
        while not terminated:

            avail_actions = self.env.get_avail_actions()
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [avail_actions],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            if use_search:
                if not search_initialised:
                    print(f"Generating trajectories, attempt {rerolls + 1}/{self.args.model_max_rerolls}")
                    H, R = self.model.mcts(self.batch, self.t_env, self.t)
                    h_index = 0
                    search_initialised = True
                    rerolls += 1

                try:
                    reward, terminated, env_info = self.env.step(H[h_index])
                except:
                    print(f"Trajectory failed at t={self.t}, h_index={h_index}")
                    # retry action with new trajectory if rerolls available
                    if rerolls < self.args.model_max_rerolls:
                        print(f"Generating trajectories, attempt {rerolls + 1}/{self.args.model_max_rerolls}")
                        H, R = self.model.mcts(self.batch, self.t_env, self.t)
                        h_index = 0
                        rerolls += 1
                        # select action, avail. actions is up to date so this will always succeed
                        reward, terminated, env_info = self.env.step(H[h_index])
                    else:
                        # no more rerolls available, default to standard search procedure
                        use_search = False

            if use_search:
                # search based action selection was successful
                actions = H[h_index].unsqueeze(0)
                partial_return += reward
                expected_partial_return += R[h_index].item()
                print(
                    f"t={self.t}: actions: {actions[0].tolist()} reward={reward:.2f}, expected reward={R[h_index].item():.2f}")
                rollout_steps += 1
                h_index += 1

            if not use_search:
                # search failed, fallback to standard action selection method
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                reward, terminated, env_info = self.env.step(actions[0])

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "battle_won": [(env_info.get("battle_won", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        if use_search:
            print(f"Search time: {time.time() - t_op_start:.2f} s")
            self.rollout_steps.append(rollout_steps)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode:
            if (len(self.test_returns) == self.args.test_nepisode):
                self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            # log rollout steps
            self.logger.log_stat("valid_rollout_steps", np.array(self.rollout_steps).mean(), self.t_env)
            self.rollout_steps = []
            self.log_train_stats_t = self.t_env

        return self.batch, episode_return, (partial_return, expected_partial_return)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
