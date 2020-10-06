REGISTRY = {}

from .basic_controller import BasicMAC
from .simple_controller import SimPLeMAC
from .model_controller import ModelMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["simple_mac"] = SimPLeMAC
REGISTRY["model_mac"] = ModelMAC