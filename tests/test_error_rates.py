"""
test_error_rates.py
-------------------
Two families of simulation-based tests:

  1. Type I error  (effect=0): false positive rate must be < FPR_THRESHOLD.
  2. Power         (effect=EFFECT_SIZE): rejection rate must be >= POWER_THRESHOLD.

Both families cover:
  - No covariate adjustment (raw pingouin baseline)
  - pfngouin + LinearModel  (runs by default)
  - pfngouin + XGBoostModel / TabPFNModel  (opt-in, -m ml_models)

All simulations use deterministic seeds (seed=i for i in range(N_SIMS)),
so test outcomes are fully reproducible.

Run default tests (LinearModel + no-adjustment):
    uv run pytest tests/test_error_rates.py -s

Run ML models as well:
    uv run pytest -m ml_models tests/test_error_rates.py -s
"""

from __future__ import annotations

import pingouin as pg
import pytest
from tqdm import tqdm

import pfngouin as pfg
from pfngouin.datasets import make_experiment_data

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ALPHA = 0.05
FPR_THRESHOLD = 0.06  # max tolerated false positive rate under H₀
POWER_THRESHOLD = 0.5  # min required rejection rate when effect is present
EFFECT_SIZE = 3.0  # additive treatment effect for power tests
N_SIMS = 500  # deterministic: seed=i for i in range(N_SIMS)
N_USERS = 200

# ---------------------------------------------------------------------------
# No-adjustment wrappers
# (match the pfngouin call signature so the shared helper works for both)
# ---------------------------------------------------------------------------


def _pg_ttest(control: object, treatment: object, **_: object) -> object:
    return pg.ttest(treatment, control)  # type: ignore[arg-type]


def _pg_mwu(control: object, treatment: object, **_: object) -> object:
    return pg.mwu(treatment, control)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_model() -> pfg.LinearModel:
    pytest.importorskip("sklearn")
    return pfg.LinearModel()


@pytest.fixture()
def xgboost_model() -> pfg.XGBoostModel:
    pytest.importorskip("xgboost")
    return pfg.XGBoostModel(tune=False)


@pytest.fixture()
def tabpfn_model() -> pfg.TabPFNModel:
    pytest.importorskip("tabpfn")
    return pfg.TabPFNModel()


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _estimate_rejection_rate(
    test_fn,
    model,
    effect: float,
    n_sims: int = N_SIMS,
) -> float:
    """
    Run *n_sims* independent experiments and return the fraction where p < ALPHA.

    *test_fn* is either a pfngouin function (pfg.ttest / pfg.mwu) or one of the
    no-adjustment wrappers (_pg_ttest / _pg_mwu) that share the same call signature.
    *model* is passed through to pfngouin; the no-adjustment wrappers ignore it.
    """
    rejections = 0
    model_name = "no adjustment" if model is None else type(model).__name__
    desc = f"{test_fn.__name__} / {model_name} / effect={effect}"
    with tqdm(range(n_sims), desc=desc, unit="sim", leave=True) as bar:
        for seed in bar:
            data = make_experiment_data(N=N_USERS, effect=effect, seed=seed)
            result = test_fn(
                data["control"],
                data["treatment"],
                covariates_control=data["X_ctrl"],
                covariates_treatment=data["X_trt"],
                model=model,
                random_state=seed,
            )
            if result["p_val"].iloc[0] < ALPHA:
                rejections += 1
            bar.set_postfix(rate=f"{rejections / (seed + 1):.3f}")
    return rejections / n_sims


# ---------------------------------------------------------------------------
# ttest — type I error  (effect = 0)
# ---------------------------------------------------------------------------


def test_ttest_type1_error_no_adjustment() -> None:
    fpr = _estimate_rejection_rate(_pg_ttest, None, effect=0)
    assert fpr < FPR_THRESHOLD, f"ttest FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


def test_ttest_type1_error_linear(linear_model: pfg.LinearModel) -> None:
    fpr = _estimate_rejection_rate(pfg.ttest, linear_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"ttest FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
def test_ttest_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate(pfg.ttest, xgboost_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"ttest FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
def test_ttest_type1_error_tabpfn(tabpfn_model: pfg.TabPFNModel) -> None:
    fpr = _estimate_rejection_rate(pfg.ttest, tabpfn_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"ttest FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# mwu — type I error  (effect = 0)
# ---------------------------------------------------------------------------


def test_mwu_type1_error_no_adjustment() -> None:
    fpr = _estimate_rejection_rate(_pg_mwu, None, effect=0)
    assert fpr < FPR_THRESHOLD, f"mwu FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


def test_mwu_type1_error_linear(linear_model: pfg.LinearModel) -> None:
    fpr = _estimate_rejection_rate(pfg.mwu, linear_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"mwu FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
def test_mwu_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate(pfg.mwu, xgboost_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"mwu FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
def test_mwu_type1_error_tabpfn(tabpfn_model: pfg.TabPFNModel) -> None:
    fpr = _estimate_rejection_rate(pfg.mwu, tabpfn_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"mwu FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# ttest — power  (effect = EFFECT_SIZE)
# ---------------------------------------------------------------------------


def test_ttest_power_no_adjustment() -> None:
    power = _estimate_rejection_rate(_pg_ttest, None, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


def test_ttest_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate(pfg.ttest, linear_model, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


@pytest.mark.ml_models
def test_ttest_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate(pfg.ttest, xgboost_model, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


@pytest.mark.ml_models
def test_ttest_power_tabpfn(tabpfn_model: pfg.TabPFNModel) -> None:
    power = _estimate_rejection_rate(pfg.ttest, tabpfn_model, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# mwu — power  (effect = EFFECT_SIZE)
# ---------------------------------------------------------------------------


def test_mwu_power_no_adjustment() -> None:
    power = _estimate_rejection_rate(_pg_mwu, None, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


def test_mwu_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate(pfg.mwu, linear_model, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


@pytest.mark.ml_models
def test_mwu_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate(pfg.mwu, xgboost_model, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"
    )


@pytest.mark.ml_models
def test_mwu_power_tabpfn(tabpfn_model: pfg.TabPFNModel) -> None:
    power = _estimate_rejection_rate(pfg.mwu, tabpfn_model, effect=EFFECT_SIZE)
    assert power >= POWER_THRESHOLD, (
        f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"
    )
