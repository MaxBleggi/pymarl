from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .simple_learner import SimPLeLearner
from .model_learner import ModelLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["simple_learner"] = SimPLeLearner
REGISTRY["mbe"] = ModelLearner
