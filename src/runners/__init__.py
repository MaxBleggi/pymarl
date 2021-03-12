REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .model_episode_runner import ModelEpisodeRunner
REGISTRY["model_episode"] = ModelEpisodeRunner

from .model_mcts_episode_runner import ModelMCTSEpisodeRunner
REGISTRY["model_mcts_episode"] = ModelMCTSEpisodeRunner

from .model_muzero_episode_runner import ModelMuZeroEpisodeRunner
REGISTRY["model_muzero_episode"] = ModelMuZeroEpisodeRunner

from .muzero_episode_runner import MuZeroEpisodeRunner
REGISTRY["muzero_episode"] = MuZeroEpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
