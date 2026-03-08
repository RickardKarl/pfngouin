from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class BaseOutcomeModel(ABC):
    #: If True, _adjust will use cross-fitting to avoid overfitting the residuals.
    crossfit_required: ClassVar[bool] = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseOutcomeModel: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
