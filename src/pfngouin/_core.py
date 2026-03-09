"""
_core.py
--------
Internal helpers for CUPED variance reduction.
Not part of the public API.
"""

from __future__ import annotations

import numpy as np

from .models.base import BaseOutcomeModel


def _validate_shapes(
    control: np.ndarray,
    treatment: np.ndarray,
    covariates_control: np.ndarray,
    covariates_treatment: np.ndarray,
) -> None:
    """Raise ValueError if outcome and covariate array lengths are inconsistent."""
    if len(control) != len(covariates_control):
        raise ValueError(
            f"control has {len(control)} rows but covariates_control has "
            f"{len(covariates_control)} rows."
        )
    if len(treatment) != len(covariates_treatment):
        raise ValueError(
            f"treatment has {len(treatment)} rows but covariates_treatment has "
            f"{len(covariates_treatment)} rows."
        )


def _crossfit_predict(
    model: BaseOutcomeModel,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int | None,
) -> np.ndarray:
    """
    Cross-validated predictions to avoid fitting on the same data used to
    adjust, preventing overfitting of the variance reduction step.
    """
    n = len(y)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n)
    folds = np.array_split(shuffled, n_splits)

    y_hat: np.ndarray = np.empty(n)
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        model.fit(X[train_idx], y[train_idx])
        y_hat[val_idx] = model.predict(X[val_idx])

    return y_hat


def _adjust(
    model: BaseOutcomeModel,
    control: np.ndarray,
    treatment: np.ndarray,
    covariates_control: np.ndarray,
    covariates_treatment: np.ndarray,
    n_splits: int = 5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit the model on the full dataset (control + treatment combined) and return
    covariate-adjusted outcomes for each group plus the variance reduction ratio.

    If model.crossfit_required is True, cross-fitting is used to prevent
    overfitting the adjustment. Otherwise, the model is fit on the full dataset
    and used to predict directly.

    Parameters
    ----------
    model                       : outcome model for variance reduction
    control, treatment          : 1-D outcome arrays
    covariates_control,
    covariates_treatment        : 2-D covariate arrays, shape (n_users, n_features)
    n_splits                    : number of cross-fitting folds (only used when
                                  model.crossfit_required is True)
    random_state                : random seed for reproducibility

    Returns
    -------
    control_adj, treatment_adj  : adjusted outcome arrays
    var_reduction               : fraction of variance explained (0–1)
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    covariates_control = np.asarray(covariates_control, dtype=float)
    covariates_treatment = np.asarray(covariates_treatment, dtype=float)

    _validate_shapes(control, treatment, covariates_control, covariates_treatment)

    X_ctrl = np.atleast_2d(covariates_control)
    X_trt = np.atleast_2d(covariates_treatment)

    # Combine across groups — theta must be estimated on the full dataset
    X = np.vstack([X_ctrl, X_trt])
    y = np.concatenate([control, treatment])

    if model.crossfit_required:
        y_pred = _crossfit_predict(model, X, y, n_splits, random_state)
    else:
        model.fit(X, y)
        y_pred = model.predict(X)

    # Traditional CUPED formula: Y_adj = Y - θ * (ŷ - E[ŷ])
    # θ = Cov(Y, ŷ) / Var(ŷ) is the OLS coefficient that minimises residual variance.
    theta = float(np.cov(y, y_pred)[0, 1] / np.var(y_pred))
    y_adj = y - theta * (y_pred - y_pred.mean())

    # Variance reduction: fraction of original variance removed by CUPED.
    var_original = float(np.var(y))
    var_adjusted = float(np.var(y_adj))
    var_reduction = float(np.clip(1.0 - var_adjusted / var_original, 0.0, 1.0))

    n_ctrl = len(control)
    control_adj = y_adj[:n_ctrl]
    treatment_adj = y_adj[n_ctrl:]

    return control_adj, treatment_adj, var_reduction
