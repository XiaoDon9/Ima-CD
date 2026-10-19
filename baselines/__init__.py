"""Source-only, DANN, Deep CORAL and MSDAL baselines."""

from .training import default_config, train
from .evaluation import evaluate

__all__ = ["default_config", "train", "evaluate"]
