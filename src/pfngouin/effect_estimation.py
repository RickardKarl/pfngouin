"""
effect_estimation.py
--------------------
AIPW-based average treatment effect (ATE) estimation for **randomised experiments**.

.. warning::
    This module is designed for data from randomised experiments.
    It is **not** suitable for observational data.  In an observational setting
    the propensity scores are unknown and must be estimated; any misspecification
    introduces confounding bias that the AIPW correction cannot fully remove.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats

from ._core import _make_folds
from .engine import InferenceEngine
from .models.base import _AnyModel


def _crossfit_aipw(
    model: _AnyModel,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    X_arr: np.ndarray,
    T_arr: np.ndarray,
    n_splits: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """K-fold cross-fitted counterfactual predictions for AIPW.

    Folds are stratified by treatment assignment (and by ``_source`` if that
    column is present in *X_df*) so each fold mirrors the overall distribution.
    Returns mu0 and mu1 assembled from out-of-fold predictions.
    """
    n = len(y_series)
    y_np = y_series.to_numpy(dtype=float)

    strat = T_arr.astype(int)
    if "_source" in X_df.columns:
        unique_sources = list(dict.fromkeys(X_df["_source"]))
        source_map = {s: i for i, s in enumerate(unique_sources)}
        src = np.array([source_map[s] for s in X_df["_source"]], dtype=int)
        strat = src * 2 + strat

    mu0 = np.empty(n, dtype=float)
    mu1 = np.empty(n, dtype=float)

    for train_idx, val_idx in _make_folds(n, n_splits, random_state, stratify_by=strat):
        m_fold = copy.deepcopy(model)

        if isinstance(m_fold, InferenceEngine):
            m_fold.fit(X_df.iloc[train_idx], y_series.iloc[train_idx])
        else:
            m_fold.fit(X_arr[train_idx], y_np[train_idx])

        X0_val = X_arr[val_idx].copy()
        X0_val[:, -1] = 0.0
        X1_val = X_arr[val_idx].copy()
        X1_val[:, -1] = 1.0
        mu0[val_idx] = np.asarray(m_fold.predict(X0_val), dtype=float)
        mu1[val_idx] = np.asarray(m_fold.predict(X1_val), dtype=float)

    return mu0, mu1


@dataclass
class ATEResult:
    """Results from an ATE estimation."""

    ate: float
    """Point estimate of the average treatment effect."""

    se: float
    """Standard error of the ATE estimate."""

    ci: tuple[float, float]
    """95% confidence interval (lower, upper)."""

    variance_reduction: float
    """Variance reduction relative to naive IPW, clipped to [0, 1]."""


def difference_in_means(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    alpha: float = 0.05,
) -> ATEResult:
    """Estimate the ATE as the simple difference in group means.

    This is the unadjusted estimator: ``E[Y | T=1] - E[Y | T=0]``.
    It is unbiased under randomisation but has no variance reduction from
    covariate adjustment.  ``variance_reduction`` is always 0.0.

    Parameters
    ----------
    data      : DataFrame containing all variables.
    outcome   : Column name of the outcome variable (Y).
    treatment : Column name of the binary treatment indicator (0/1).
    alpha     : significance level for the confidence interval (default 0.05).

    Returns
    -------
    ATEResult with fields: ate, se, ci, variance_reduction
    """
    Y = data[outcome].to_numpy(dtype=float)
    T = data[treatment].to_numpy(dtype=float)

    y1 = Y[T == 1]
    y0 = Y[T == 0]
    n1, n0 = len(y1), len(y0)

    ate = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0))
    z = float(scipy.stats.norm.ppf(1 - alpha / 2))
    ci: tuple[float, float] = (ate - z * se, ate + z * se)

    return ATEResult(ate=ate, se=se, ci=ci, variance_reduction=0.0)


def aipw(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: str | list[str],
    model: _AnyModel | None = None,
    alpha: float = 0.05,
    propensity_score: float | np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
) -> ATEResult:
    """Estimate the ATE using Augmented Inverse Probability Weighting (AIPW).

    .. note::
        This is an AIPW (Augmented Inverse Probability Weighting) estimator
        designed for **randomised experiments** only.
        It assumes unconfounded treatment assignment.  Do not use it on
        observational data where treatment selection may depend on covariates
        in ways that are not fully captured by the propensity score.

    Parameters
    ----------
    data             : DataFrame containing all variables.
    outcome          : Column name of the outcome variable (Y).
    treatment        : Column name of the binary treatment indicator (0/1).
    covariates       : Column name(s) to use as covariates (X). Accepts a
                       single string or a list of strings.
    model            : outcome model (any BaseOutcomeModel or InferenceEngine)
    alpha            : significance level for the confidence interval (default 0.05)
    propensity_score : known propensity score(s) from the experimental design.
                       Can be a scalar (same value for all units, e.g. 0.5 for
                       a balanced RCT) or a 1-D array of per-unit scores,
                       shape (n,).  When ``None`` (default) the empirical
                       treatment rate is used, which is the correct estimator
                       for simple randomised experiments.

    Returns
    -------
    ATEResult with fields: ate, se, ci, variance_reduction
    """
    if model is None:
        from .models.pfn import PFNModel

        model = PFNModel()

    cov_list: list[str] = (
        [covariates] if isinstance(covariates, str) else list(covariates)
    )

    feature_cols = cov_list + [treatment]
    X_df = data[feature_cols].copy()
    y_series = data[outcome].copy()  # Series.name == outcome already

    Y_arr = y_series.to_numpy(dtype=float)
    T_arr = data[treatment].to_numpy(dtype=float)
    n = len(Y_arr)

    e: float | np.ndarray
    if propensity_score is None:
        e = float(T_arr.mean())  # empirical propensity (valid for RCTs)
    else:
        e = np.asarray(propensity_score, dtype=float)
        if e.ndim == 0:
            e = float(e)
        elif e.shape != (n,):
            raise ValueError(
                f"propensity_score array must have shape ({n},), got {e.shape}"
            )

    # Predict counterfactuals (treatment col is last → index -1)
    X_arr = X_df.to_numpy(dtype=float)

    if model.crossfit_required:
        mu0, mu1 = _crossfit_aipw(
            model, X_df, y_series, X_arr, T_arr, n_splits, random_state
        )
    else:
        m = copy.deepcopy(model)
        if isinstance(m, InferenceEngine):
            m.fit(X_df, y_series)
        else:
            m.fit(X_arr, Y_arr)
        X0 = X_arr.copy()
        X0[:, -1] = 0.0
        X1 = X_arr.copy()
        X1[:, -1] = 1.0
        mu0 = np.asarray(m.predict(X0), dtype=float)
        mu1 = np.asarray(m.predict(X1), dtype=float)

    # AIPW influence-function scores
    psi = mu1 - mu0 + T_arr * (Y_arr - mu1) / e - (1 - T_arr) * (Y_arr - mu0) / (1 - e)

    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    z = float(scipy.stats.norm.ppf(1 - alpha / 2))
    ci: tuple[float, float] = (ate - z * se, ate + z * se)

    # Variance reduction vs naive IPW (no outcome model)
    naive = T_arr * Y_arr / e - (1 - T_arr) * Y_arr / (1 - e)
    var_psi = float(np.var(psi, ddof=1))
    var_naive = float(np.var(naive, ddof=1))
    variance_reduction = float(np.clip(1.0 - var_psi / var_naive, 0.0, 1.0))

    return ATEResult(
        ate=ate,
        se=se,
        ci=ci,
        variance_reduction=variance_reduction,
    )
