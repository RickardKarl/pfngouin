"""
Tests for effect_estimation.py (ATEResult and aipw).
"""

from __future__ import annotations

import pandas as pd
from _data import make_experiment_data

from pfngouin import ATEResult, aipw
from pfngouin.models import LinearModel


def _prepare(N: int = 200, effect: float = 1.0, seed: int = 42) -> pd.DataFrame:
    df = make_experiment_data(N=N, effect=effect, seed=seed)
    return df.assign(treated=(df["group"] == "treatment").astype(float))


_COVARIATES = ["pre1", "pre2", "pre3"]


# ---------------------------------------------------------------------------
# ATEResult / aipw
# ---------------------------------------------------------------------------


def test_aipw_result_fields() -> None:
    df = _prepare()
    result = aipw(
        df,
        outcome="outcome",
        treatment="treated",
        covariates=_COVARIATES,
        model=LinearModel(),
    )
    assert isinstance(result, ATEResult)
    assert isinstance(result.ate, float)
    assert isinstance(result.ci, tuple) and len(result.ci) == 2
    assert isinstance(result.variance_reduction, float)


def test_variance_reduction_range() -> None:
    df = _prepare()
    result = aipw(
        df,
        outcome="outcome",
        treatment="treated",
        covariates=_COVARIATES,
        model=LinearModel(),
    )
    assert 0.0 <= result.variance_reduction <= 1.0


def test_ci_contains_true_effect() -> None:
    true_ate = 2.0
    df = _prepare(N=500, effect=true_ate, seed=7)
    result = aipw(
        df,
        outcome="outcome",
        treatment="treated",
        covariates=_COVARIATES,
        model=LinearModel(),
    )
    assert result.ci[0] <= true_ate <= result.ci[1]


def test_aipw_string_covariates() -> None:
    df = _prepare()
    result = aipw(
        df,
        outcome="outcome",
        treatment="treated",
        covariates="pre1",
        model=LinearModel(),
    )
    assert isinstance(result, ATEResult)
    assert 0.0 <= result.variance_reduction <= 1.0


def test_aipw_crossfit() -> None:
    """crossfit_required=True triggers K-fold cross-fitting; result is still valid."""

    class CrossfitLinear(LinearModel):
        crossfit_required: bool = True  # type: ignore[assignment]

    df = _prepare(N=300, effect=1.0, seed=99)
    result = aipw(
        df,
        outcome="outcome",
        treatment="treated",
        covariates=_COVARIATES,
        model=CrossfitLinear(),
        n_splits=3,
        random_state=0,
    )
    assert isinstance(result, ATEResult)
    assert 0.0 <= result.variance_reduction <= 1.0
    assert result.ci[0] < result.ci[1]
