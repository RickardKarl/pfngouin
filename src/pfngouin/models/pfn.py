"""
pfn.py
------
Unified PFN outcome model supporting two backends:
  - "tabpfn"  : TabPFNRegressor via tabpfn-client (cloud API)
  - "tabicl"  : TabICLRegressor (local, sklearn-compatible)
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from .base import BaseOutcomeModel


class PFNModel(BaseOutcomeModel):
    """Foundation-model outcome model with selectable backend.

    Parameters
    ----------
    backend : "tabicl" (default) or "tabpfn-client"
    **kwargs : passed through to the underlying regressor constructor
    """

    crossfit_required: ClassVar[bool] = True

    def __init__(
        self,
        backend: Literal["tabpfn-client", "tabicl"] = "tabicl",
        **kwargs: object,
    ) -> None:
        if backend not in ("tabpfn-client", "tabicl"):
            raise ValueError(
                f"backend must be 'tabpfn-client' or 'tabicl', got {backend!r}"
            )
        self._backend = backend
        self._model: object = None

        if backend == "tabpfn-client":
            try:
                from tabpfn_client import TabPFNRegressor  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "PFNModel(backend='tabpfn-client') requires tabpfn-client."
                ) from exc
            self._model = TabPFNRegressor(**kwargs)
        elif backend == "tabicl":
            try:
                from tabicl import TabICLRegressor  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "PFNModel(backend='tabicl') requires tabicl."
                ) from exc
            self._model = TabICLRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> PFNModel:
        self._model.fit(X, y)  # type: ignore[union-attr]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = self._model.predict(X)  # type: ignore[union-attr]
        return np.asarray(result, dtype=float)


class TabPFNModel(PFNModel):
    """Backward-compatible alias for PFNModel(backend='tabpfn-client')."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(backend="tabpfn-client", **kwargs)


class TabICLModel(PFNModel):
    """Backward-compatible alias for PFNModel(backend='tabicl')."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(backend="tabicl", **kwargs)
