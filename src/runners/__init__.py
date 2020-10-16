REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .model_episode_runner import ModelEpisodeRunner
REGISTRY["model_episode"] = ModelEpisodeRunner

from .model_mcts_episode_runner import ModelMCTSEpisodeRunner
REGISTRY["model_mcts_episode"] = ModelMCTSEpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
