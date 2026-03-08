from __future__ import annotations

from typing import ClassVar

import numpy as np

from .base import BaseOutcomeModel


class LinearModel(BaseOutcomeModel):
    """Outcome model backed by scikit-learn's LinearRegression.

    Requires scikit-learn: ``pip install scikit-learn``.
    """

    crossfit_required: ClassVar[bool] = False

    def __init__(self, **kwargs: object) -> None:
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("LinearModel requires scikit-learn. ") from exc
        self._model = LinearRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearModel:
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=float)
