__version__ = "0.1.0"

from .datasets import make_experiment_data
from .inference import mwu, ttest
from .models import BaseOutcomeModel, LinearModel, TabPFNModel, XGBoostModel

__all__ = [
    "ttest",
    "mwu",
    "make_experiment_data",
    "BaseOutcomeModel",
    "LinearModel",
    "TabPFNModel",
    "XGBoostModel",
    "__version__",
]
