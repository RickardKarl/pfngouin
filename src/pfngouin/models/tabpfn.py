from __future__ import annotations

import contextlib
import io
from typing import ClassVar

import numpy as np

from .base import BaseOutcomeModel


class TabPFNModel(BaseOutcomeModel):
    """Outcome model backed by TabPFNRegressor (cloud API via tabpfn-client).

    Requires the `tabpfn` extra: ``pip install pfngouin[tabpfn]``.
    Authentication is handled by tabpfn-client on first use.
    """

    crossfit_required: ClassVar[bool] = True

    def __init__(self, **tabpfn_kwargs: object) -> None:
        try:
            from tabpfn_client import TabPFNRegressor  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("TabPFNModel requires tabpfn-client. ") from exc
        self._model = TabPFNRegressor(**tabpfn_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabPFNModel:
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with contextlib.redirect_stderr(io.StringIO()):
            result = self._model.predict(X)
        return np.asarray(result, dtype=float)
