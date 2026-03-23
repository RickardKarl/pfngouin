"""
test_hypothesis_testing_simulations.py
---------------------------------------
Two families of simulation-based tests for hypothesis_testing.py:

  1. Type I error  (effect=0): false positive rate must be < FPR_THRESHOLD.
  2. Power         (effect=EFFECT_SIZE): rejection rate must be >= POWER_THRESHOLD.

Both families cover:
  - No covariate adjustment (raw pingouin baseline)
  - pfngouin + LinearModel  (runs by default)
  - pfngouin + XGBoostModel / PFNModel  (opt-in, -m ml_models)

All simulations use deterministic seeds (seed=i for i in range(N_SIMS)),
so test outcomes are fully reproducible.

Run default tests (LinearModel + no-adjustment):
    uv run pytest tests/test_hypothesis_testing_simulations.py -s

Run ML models as well:
    uv run pytest -m ml_models tests/test_hypothesis_testing_simulations.py -s
"""

from __future__ import annotations

import pingouin as pg
import pytest

pytestmark = pytest.mark.simulations
from tqdm import tqdm

import pfngouin as pfg
from _data import make_experiment_data, make_multigroup_df

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ALPHA = 0.05
FPR_THRESHOLD = 0.06  # max tolerated false positive rate under H₀
POWER_THRESHOLD = 0.5  # min required rejection rate when effect is present
EFFECT_SIZE = 3.0  # additive treatment effect for power tests
N_SIMS = 1000  # deterministic: seed=i for i in range(N_SIMS)
N_USERS = 2500

TOST_BOUND = 1.0  # equivalence bound for all tost tests
N_USERS_MULTI = 300  # total users for 3-group tests (100 per group)
MULTI_EFFECTS_NULL = (0.0, 0.0, 0.0)  # all groups equal (type I error)
MULTI_EFFECTS_POWER = (0.0, 0.0, EFFECT_SIZE)  # one group different (power)
COVAR_COLS = ["pre1", "pre2", "pre3"]

# ---------------------------------------------------------------------------
# No-adjustment wrappers
# (match the pfngouin call signature so the shared helper works for both)
# ---------------------------------------------------------------------------


def _pg_ttest(
    data: object,
    dv: object,
    between: object,
    **_: object,
) -> object:
    import pandas as pd
    df = data  # type: ignore[assignment]
    groups = sorted(df[between].unique())  # type: ignore[index]
    x = df.loc[df[between] == groups[1], dv]  # type: ignore[index]
    y = df.loc[df[between] == groups[0], dv]  # type: ignore[index]
    return pg.ttest(x, y)  # type: ignore[arg-type]


def _pg_mwu(
    data: object,
    dv: object,
    between: object,
    **_: object,
) -> object:
    df = data  # type: ignore[assignment]
    groups = sorted(df[between].unique())  # type: ignore[index]
    x = df.loc[df[between] == groups[1], dv]  # type: ignore[index]
    y = df.loc[df[between] == groups[0], dv]  # type: ignore[index]
    return pg.mwu(x, y)  # type: ignore[arg-type]


def _pg_tost(
    data: object,
    dv: object,
    between: object,
    **_: object,
) -> object:
    df = data  # type: ignore[assignment]
    groups = sorted(df[between].unique())  # type: ignore[index]
    x = df.loc[df[between] == groups[1], dv]  # type: ignore[index]
    y = df.loc[df[between] == groups[0], dv]  # type: ignore[index]
    result = pg.tost(x, y, bound=TOST_BOUND)  # type: ignore[arg-type]
    return result.rename(columns={"pval": "p_val"})


def _pfg_tost(
    data: object,
    dv: object,
    between: object,
    covar: object = None,
    model: object = None,
    random_state: object = None,
    **_: object,
) -> object:
    result = pfg.tost(  # type: ignore[arg-type]
        data,  # type: ignore[arg-type]
        dv=dv,  # type: ignore[arg-type]
        between=between,  # type: ignore[arg-type]
        covar=covar,  # type: ignore[arg-type]
        bound=TOST_BOUND,
        model=model,  # type: ignore[arg-type]
        random_state=random_state,  # type: ignore[arg-type]
    )
    return result.rename(columns={"pval": "p_val"})


def _pg_anova(
    data: object,
    dv: object,
    between: object,
    covar: object = None,
    model: object = None,
    random_state: object = None,
    **_: object,
) -> object:
    return pg.anova(data=data, dv=dv, between=between)  # type: ignore[arg-type]


def _pg_welch_anova(
    data: object,
    dv: object,
    between: object,
    covar: object = None,
    model: object = None,
    random_state: object = None,
    **_: object,
) -> object:
    return pg.welch_anova(data=data, dv=dv, between=between)  # type: ignore[arg-type]


def _pg_kruskal(
    data: object,
    dv: object,
    between: object,
    covar: object = None,
    model: object = None,
    random_state: object = None,
    **_: object,
) -> object:
    return pg.kruskal(data=data, dv=dv, between=between)  # type: ignore[arg-type]


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
                data=data,
                dv="outcome",
                between="group",
                covar=COVAR_COLS,
                model=model,
                random_state=seed,
            )
            if result["p_val"].iloc[0] < ALPHA:
                rejections += 1
            bar.set_postfix(rate=f"{rejections / (seed + 1):.3f}")
    return rejections / n_sims


def _estimate_rejection_rate_multi_group(
    test_fn: object,
    model: object,
    effects: tuple[float, ...],
    pval_col: str = "p_unc",
    n_sims: int = N_SIMS,
) -> float:
    """Like _estimate_rejection_rate but for DataFrame-based multi-group tests."""
    rejections = 0
    model_name = "no adjustment" if model is None else type(model).__name__
    desc = f"{test_fn.__name__} / {model_name} / effects={effects}"  # type: ignore[union-attr]
    with tqdm(range(n_sims), desc=desc, unit="sim", leave=True) as bar:
        for seed in bar:
            df = make_multigroup_df(N=N_USERS_MULTI, effects=effects, seed=seed)
            result = test_fn(  # type: ignore[operator]
                data=df,
                dv="outcome",
                between="group",
                covar=COVAR_COLS,
                model=model,
                random_state=seed,
            )
            if result[pval_col].iloc[0] < ALPHA:  # type: ignore[index]
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
@pytest.mark.xgboost
def test_ttest_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate(pfg.ttest, xgboost_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"ttest FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_ttest_type1_error_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
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
@pytest.mark.xgboost
def test_mwu_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate(pfg.mwu, xgboost_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"mwu FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_mwu_type1_error_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    fpr = _estimate_rejection_rate(pfg.mwu, tabpfn_model, effect=0)
    assert fpr < FPR_THRESHOLD, f"mwu FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# ttest — power  (effect = EFFECT_SIZE)
# ---------------------------------------------------------------------------


def test_ttest_power_no_adjustment() -> None:
    power = _estimate_rejection_rate(_pg_ttest, None, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"


def test_ttest_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate(pfg.ttest, linear_model, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_ttest_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate(pfg.ttest, xgboost_model, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_ttest_power_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    power = _estimate_rejection_rate(pfg.ttest, tabpfn_model, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"ttest power={power:.3f} below threshold {POWER_THRESHOLD}"


# ---------------------------------------------------------------------------
# mwu — power  (effect = EFFECT_SIZE)
# ---------------------------------------------------------------------------


def test_mwu_power_no_adjustment() -> None:
    power = _estimate_rejection_rate(_pg_mwu, None, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"


def test_mwu_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate(pfg.mwu, linear_model, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_mwu_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate(pfg.mwu, xgboost_model, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_mwu_power_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    power = _estimate_rejection_rate(pfg.mwu, tabpfn_model, effect=EFFECT_SIZE)
    assert (
        power >= POWER_THRESHOLD
    ), f"mwu power={power:.3f} below threshold {POWER_THRESHOLD}"


# ---------------------------------------------------------------------------
# tost — type I error  (effect = TOST_BOUND)
# ---------------------------------------------------------------------------


def test_tost_type1_error_no_adjustment() -> None:
    fpr = _estimate_rejection_rate(_pg_tost, None, effect=EFFECT_SIZE)
    assert fpr < FPR_THRESHOLD, f"tost FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


def test_tost_type1_error_linear(linear_model: pfg.LinearModel) -> None:
    fpr = _estimate_rejection_rate(_pfg_tost, linear_model, effect=EFFECT_SIZE)
    assert fpr < FPR_THRESHOLD, f"tost FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_tost_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate(_pfg_tost, xgboost_model, effect=EFFECT_SIZE)
    assert fpr < FPR_THRESHOLD, f"tost FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_tost_type1_error_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    fpr = _estimate_rejection_rate(_pfg_tost, tabpfn_model, effect=EFFECT_SIZE)
    assert fpr < FPR_THRESHOLD, f"tost FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# tost — power  (effect = 0 < TOST_BOUND → truly equivalent)
# ---------------------------------------------------------------------------


def test_tost_power_no_adjustment() -> None:
    power = _estimate_rejection_rate(_pg_tost, None, effect=0)
    assert (
        power >= POWER_THRESHOLD
    ), f"tost power={power:.3f} below threshold {POWER_THRESHOLD}"


def test_tost_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate(_pfg_tost, linear_model, effect=0)
    assert (
        power >= POWER_THRESHOLD
    ), f"tost power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_tost_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate(_pfg_tost, xgboost_model, effect=0)
    assert (
        power >= POWER_THRESHOLD
    ), f"tost power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_tost_power_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    power = _estimate_rejection_rate(_pfg_tost, tabpfn_model, effect=0)
    assert (
        power >= POWER_THRESHOLD
    ), f"tost power={power:.3f} below threshold {POWER_THRESHOLD}"


# ---------------------------------------------------------------------------
# anova — type I error  (all groups equal)
# ---------------------------------------------------------------------------


def test_anova_type1_error_no_adjustment() -> None:
    fpr = _estimate_rejection_rate_multi_group(_pg_anova, None, MULTI_EFFECTS_NULL)
    assert fpr < FPR_THRESHOLD, f"anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


def test_anova_type1_error_linear(linear_model: pfg.LinearModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.anova, linear_model, MULTI_EFFECTS_NULL
    )
    assert fpr < FPR_THRESHOLD, f"anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_anova_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.anova, xgboost_model, MULTI_EFFECTS_NULL
    )
    assert fpr < FPR_THRESHOLD, f"anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_anova_type1_error_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.anova, tabpfn_model, MULTI_EFFECTS_NULL
    )
    assert fpr < FPR_THRESHOLD, f"anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# anova — power  (one group different)
# ---------------------------------------------------------------------------


def test_anova_power_no_adjustment() -> None:
    power = _estimate_rejection_rate_multi_group(_pg_anova, None, MULTI_EFFECTS_POWER)
    assert (
        power >= POWER_THRESHOLD
    ), f"anova power={power:.3f} below threshold {POWER_THRESHOLD}"


def test_anova_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.anova, linear_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"anova power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_anova_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.anova, xgboost_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"anova power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_anova_power_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.anova, tabpfn_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"anova power={power:.3f} below threshold {POWER_THRESHOLD}"


# ---------------------------------------------------------------------------
# welch_anova — type I error  (all groups equal)
# ---------------------------------------------------------------------------


def test_welch_anova_type1_error_no_adjustment() -> None:
    fpr = _estimate_rejection_rate_multi_group(
        _pg_welch_anova, None, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"welch_anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


def test_welch_anova_type1_error_linear(linear_model: pfg.LinearModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.welch_anova, linear_model, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"welch_anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_welch_anova_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.welch_anova, xgboost_model, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"welch_anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_welch_anova_type1_error_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.welch_anova, tabpfn_model, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"welch_anova FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# welch_anova — power  (one group different)
# ---------------------------------------------------------------------------


def test_welch_anova_power_no_adjustment() -> None:
    power = _estimate_rejection_rate_multi_group(
        _pg_welch_anova, None, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"welch_anova power={power:.3f} below threshold {POWER_THRESHOLD}"


def test_welch_anova_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.welch_anova, linear_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"welch_anova power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_welch_anova_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.welch_anova, xgboost_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"welch_anova power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_welch_anova_power_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.welch_anova, tabpfn_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"welch_anova power={power:.3f} below threshold {POWER_THRESHOLD}"


# ---------------------------------------------------------------------------
# kruskal — type I error  (all groups equal)
# ---------------------------------------------------------------------------


def test_kruskal_type1_error_no_adjustment() -> None:
    fpr = _estimate_rejection_rate_multi_group(_pg_kruskal, None, MULTI_EFFECTS_NULL)
    assert (
        fpr < FPR_THRESHOLD
    ), f"kruskal FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


def test_kruskal_type1_error_linear(linear_model: pfg.LinearModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.kruskal, linear_model, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"kruskal FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_kruskal_type1_error_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.kruskal, xgboost_model, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"kruskal FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_kruskal_type1_error_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    fpr = _estimate_rejection_rate_multi_group(
        pfg.kruskal, tabpfn_model, MULTI_EFFECTS_NULL
    )
    assert (
        fpr < FPR_THRESHOLD
    ), f"kruskal FPR={fpr:.3f} exceeds threshold {FPR_THRESHOLD}"


# ---------------------------------------------------------------------------
# kruskal — power  (one group different)
# ---------------------------------------------------------------------------


def test_kruskal_power_no_adjustment() -> None:
    power = _estimate_rejection_rate_multi_group(_pg_kruskal, None, MULTI_EFFECTS_POWER)
    assert (
        power >= POWER_THRESHOLD
    ), f"kruskal power={power:.3f} below threshold {POWER_THRESHOLD}"


def test_kruskal_power_linear(linear_model: pfg.LinearModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.kruskal, linear_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"kruskal power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_kruskal_power_xgboost(xgboost_model: pfg.XGBoostModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.kruskal, xgboost_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"kruskal power={power:.3f} below threshold {POWER_THRESHOLD}"


@pytest.mark.ml_models
@pytest.mark.pfn
def test_kruskal_power_tabpfn(tabpfn_model: pfg.PFNModel) -> None:
    power = _estimate_rejection_rate_multi_group(
        pfg.kruskal, tabpfn_model, MULTI_EFFECTS_POWER
    )
    assert (
        power >= POWER_THRESHOLD
    ), f"kruskal power={power:.3f} below threshold {POWER_THRESHOLD}"
