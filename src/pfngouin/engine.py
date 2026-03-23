"""
engine.py
---------
InferenceEngine: wraps any BaseOutcomeModel with optional ContextStore
context augmentation. Duck-types as a model (fit/predict/crossfit_required)
so it can be dropped in wherever a model is accepted.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from .context import ContextStore
from .models.base import BaseOutcomeModel


class InferenceEngine:
    """Wraps a BaseOutcomeModel with optional ContextStore context augmentation.

    When fitting, prepends context rows from the store that share the same
    outcome column as the training target. Columns absent in a given context
    DataFrame are NaN-filled so heterogeneous experiments can all contribute.

    A ``_source`` column is appended to the feature matrix so the model can
    learn experiment-specific effects.  Context rows receive integer codes
    1, 2, … (one per unique experiment name); the target data receives code 0.
    ``predict()`` automatically appends code 0 to any input.

    Parameters
    ----------
    model         : outcome model to wrap
    context_store : store of prior DataFrames (required)
    """

    def __init__(self, model: BaseOutcomeModel, context_store: ContextStore) -> None:
        self.model = model
        self.context_store = context_store
        self._fitted_model: BaseOutcomeModel | None = None
        self._feature_cols: list[str] = []
        self._outcome_col: str | None = None
        self._has_source_feature: bool = False

    @property
    def crossfit_required(self) -> bool:
        return self.model.crossfit_required

    def fit(self, X: pd.DataFrame, y: pd.Series) -> InferenceEngine:
        self._feature_cols = list(X.columns)
        if not self._feature_cols:
            raise ValueError("fit() called with a DataFrame that has no columns.")
        if y.name is None:
            raise ValueError("fit() called with an unnamed Series (y.name is None).")
        self._outcome_col = str(y.name)
        X_np, y_np = self._augment_with_context(
            X.to_numpy(dtype=float), y.to_numpy(dtype=float)
        )
        self._fitted_model = copy.deepcopy(self.model)
        self._fitted_model.fit(X_np, y_np)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._fitted_model is None:
            raise RuntimeError(
                "InferenceEngine must be fitted before calling predict()"
            )
        if self._has_source_feature:
            X = np.hstack([X, np.zeros((len(X), 1), dtype=float)])
        return self._fitted_model.predict(X)

    def _augment_with_context(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ctx_df = self.context_store.get_context()
        if self._outcome_col not in ctx_df.columns:
            warnings.warn(
                f"No context experiment contains outcome column '{self._outcome_col}'; "
                "context augmentation will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            return X, y

        ctx_df = ctx_df[ctx_df[self._outcome_col].notna()]
        if ctx_df.empty:
            warnings.warn(
                f"All context rows for outcome column '{self._outcome_col}' are NaN; "
                "context augmentation will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            return X, y

        X_ctx = ctx_df.reindex(columns=self._feature_cols).to_numpy(dtype=float)
        y_ctx = ctx_df[self._outcome_col].to_numpy(dtype=float)

        # Label-encode _source: target = 0, each experiment = 1, 2, ...
        unique_sources = list(dict.fromkeys(ctx_df["_source"]))  # stable order
        source_map = {s: float(i + 1) for i, s in enumerate(unique_sources)}
        source_ctx = np.array(
            [source_map[s] for s in ctx_df["_source"]], dtype=float
        ).reshape(-1, 1)
        source_target = np.zeros((len(X), 1), dtype=float)

        self._has_source_feature = True
        return (
            np.vstack(
                [
                    np.hstack([X_ctx, source_ctx]),
                    np.hstack([X, source_target]),
                ]
            ),
            np.concatenate([y_ctx, y]),
        )
