from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .simple_learner import SimPLeLearner
from .model_learner import ModelLearner
from .model_mcts_learner import ModelMCTSLearner
from .model_muzero_learner import ModelMuZeroLearner
from .muzero_learner import MuZeroLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["simple_learner"] = SimPLeLearner
REGISTRY["mbe"] = ModelLearner
REGISTRY["model_mcts"] = ModelMCTSLearner
REGISTRY["model_muzero"] = ModelMuZeroLearner
REGISTRY["muzero"] = ModelMuZeroLearner
