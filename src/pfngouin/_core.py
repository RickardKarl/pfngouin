"""
_core.py
--------
Internal helpers for CUPED variance reduction.
Not part of the public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import InferenceEngine
from .models.base import _AnyModel


def _make_folds(
    n: int,
    n_splits: int,
    random_state: int | None,
    stratify_by: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, val_idx) pairs for K-fold cross-validation.

    When *stratify_by* is provided (integer class labels, shape ``(n,)``), folds
    are balanced across classes using :class:`sklearn.model_selection.StratifiedKFold`.
    Falls back to random shuffling if stratification fails (e.g. too few samples
    per class).
    """
    if stratify_by is not None:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state if random_state is not None else 0,
        )
        try:
            return [(train, val) for train, val in skf.split(np.arange(n), stratify_by)]
        except ValueError:
            pass  # fall through to non-stratified

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n)
    raw_folds = np.array_split(shuffled, n_splits)
    return [
        (
            np.concatenate([raw_folds[j] for j in range(n_splits) if j != i]),
            raw_folds[i],
        )
        for i in range(n_splits)
    ]


def _crossfit_predict(
    model: _AnyModel,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int,
    random_state: int | None,
) -> np.ndarray:
    """
    Cross-validated predictions to avoid fitting on the same data used to
    adjust, preventing overfitting of the variance reduction step.
    """
    n = len(y)
    y_hat: np.ndarray = np.empty(n)
    use_df = isinstance(model, InferenceEngine) and isinstance(X, pd.DataFrame)
    if not use_df:
        X_np = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else y
    for train_idx, val_idx in _make_folds(n, n_splits, random_state):
        if use_df:
            X_train = X.iloc[train_idx]
            X_val_np = X.iloc[val_idx].to_numpy(dtype=float)
            y_train = y.iloc[train_idx]
        else:
            X_train = X_np[train_idx]
            X_val_np = X_np[val_idx]
            y_train = y_arr[train_idx]
        model.fit(X_train, y_train)
        y_hat[val_idx] = model.predict(X_val_np)
    return y_hat


def _cuped_core(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model: _AnyModel,
    n_splits: int,
    random_state: int | None,
) -> tuple[np.ndarray, float]:
    """
    Fit model on (X, y) and return CUPED-adjusted outcomes + variance reduction ratio.

    This is the single source of truth for the CUPED formula used by _adjust_df.
    """
    if model.crossfit_required:
        y_pred = _crossfit_predict(model, X, y, n_splits, random_state)
    else:
        if isinstance(model, InferenceEngine) and isinstance(X, pd.DataFrame):
            model.fit(X, y)
            X_np = X.to_numpy(dtype=float)
        else:
            X_np = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X
            y_fit = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else y
            model.fit(X_np, y_fit)
        y_pred = model.predict(X_np)

    y_vals = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float)

    # Traditional CUPED formula: Y_adj = Y - θ * (ŷ - E[ŷ])
    # θ = Cov(Y, ŷ) / Var(ŷ) is the OLS coefficient that minimises residual variance.
    theta = float(np.cov(y_vals, y_pred)[0, 1] / np.var(y_pred))
    y_adj = y_vals - theta * (y_pred - y_pred.mean())

    # Variance reduction: fraction of original variance removed by CUPED.
    var_original = float(np.var(y_vals))
    var_adjusted = float(np.var(y_adj))
    var_reduction = float(np.clip(1.0 - var_adjusted / var_original, 0.0, 1.0))

    return np.asarray(y_adj), var_reduction


def _adjust_df(
    data: pd.DataFrame,
    dv: str,
    covar: str | list[str],
    model: _AnyModel,
    n_splits: int = 5,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    CUPED adjustment for DataFrame input (used by all tests).

    Fits the model on the full dataset (all groups combined) and returns a
    copy of `data` with the `dv` column replaced by CUPED-adjusted values,
    plus the variance reduction ratio.

    Parameters
    ----------
    data         : DataFrame containing outcome and covariate columns
    dv           : name of the dependent variable (outcome) column
    covar        : covariate column name(s) for CUPED adjustment
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (used when model.crossfit_required)
    random_state : random seed for reproducibility

    Returns
    -------
    adjusted_data : copy of `data` with the `dv` column replaced by adjusted values
    var_reduction : fraction of variance explained (0–1)
    """
    covar_cols = [covar] if isinstance(covar, str) else list(covar)
    X = data[covar_cols]  # pd.DataFrame, column names preserved
    y = data[dv]           # pd.Series with name=dv

    y_adj, var_reduction = _cuped_core(X, y, model, n_splits, random_state)

    adjusted_data = data.copy()
    adjusted_data[dv] = y_adj
    return adjusted_data, var_reduction
