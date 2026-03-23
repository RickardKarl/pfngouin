from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Union

import numpy as np

if TYPE_CHECKING:
    from pfngouin.engine import InferenceEngine


class BaseOutcomeModel(ABC):
    #: If True, _adjust will use cross-fitting to avoid overfitting the residuals.
    crossfit_required: ClassVar[bool] = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseOutcomeModel: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# Shared type alias used across _core, hypothesis_testing, and effect_estimation.
_AnyModel = Union[BaseOutcomeModel, "InferenceEngine"]
