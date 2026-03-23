"""
test_effect_estimation_simulations.py
--------------------------------------
Simulation-based tests for the AIPW ATE estimator (effect_estimation.py):

  1. Bias: |mean(ATE_hat) - true_effect| < BIAS_SIGMAS Monte Carlo SEs
  2. Coverage: fraction of nominal 95% CIs containing the true effect ≈ 95%

Tests run for LinearModel by default; XGBoost and TabPFN are opt-in:
    uv run pytest tests/test_effect_estimation_simulations.py -s
    uv run pytest -m ml_models tests/test_effect_estimation_simulations.py -s

All simulations use deterministic seeds (seed=i for i in range(N_SIMS)).
"""

from __future__ import annotations

import numpy as np
import pytest
from _data import make_experiment_data
from tqdm import tqdm

import pfngouin as pfg

pytestmark = pytest.mark.simulations


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

TRUE_EFFECT = 2.0  # additive treatment effect applied in each simulation
N_SIMS = 1000  # Monte Carlo replications (deterministic: seed = i)
N_USERS = 500  # users per simulation
NOMINAL_COVERAGE = 0.95  # target CI coverage
COVERAGE_TOLERANCE = 0.04  # max tolerated |coverage - nominal| (≈ 4 MC SEs)
BIAS_SIGMAS = 2.0  # bias must be < this many Monte Carlo SEs


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------


def _run_dim_simulations() -> dict[str, np.ndarray]:
    """
    Run N_SIMS replications of difference-in-means
    and return ate estimates and CI hit indicators.
    """
    ates = np.empty(N_SIMS)
    hits = np.empty(N_SIMS, dtype=bool)

    for seed in tqdm(range(N_SIMS), desc="DifferenceInMeans", unit="sim", leave=True):
        df = make_experiment_data(N=N_USERS, effect=TRUE_EFFECT, seed=seed)
        df = df.assign(treated=(df["group"] == "treatment").astype(float))

        result = pfg.difference_in_means(
            df,
            outcome="outcome",
            treatment="treated",
        )
        ates[seed] = result.ate
        hits[seed] = result.ci[0] <= TRUE_EFFECT <= result.ci[1]

    return {"ates": ates, "hits": hits}


def _run_simulations(model: object) -> dict[str, np.ndarray]:
    """Run N_SIMS replications and return ate estimates and CI hit indicators."""
    model_name = type(model).__name__
    ates = np.empty(N_SIMS)
    hits = np.empty(N_SIMS, dtype=bool)

    for seed in tqdm(
        range(N_SIMS), desc=f"AIPW / {model_name}", unit="sim", leave=True
    ):
        df = make_experiment_data(N=N_USERS, effect=TRUE_EFFECT, seed=seed)
        df = df.assign(treated=(df["group"] == "treatment").astype(float))

        result = pfg.aipw(
            df,
            outcome="outcome",
            treatment="treated",
            covariates=["pre1", "pre2", "pre3"],
            model=model,
            random_state=seed,
        )
        ates[seed] = result.ate
        hits[seed] = result.ci[0] <= TRUE_EFFECT <= result.ci[1]

    return {"ates": ates, "hits": hits}


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_bias(results: dict[str, np.ndarray], label: str) -> None:
    ates = results["ates"]
    bias = ates.mean() - TRUE_EFFECT
    mcse = ates.std(ddof=1) / np.sqrt(N_SIMS)
    assert abs(bias) < BIAS_SIGMAS * mcse, (
        f"{label} bias={bias:.4f} exceeds {BIAS_SIGMAS} MCSE ({BIAS_SIGMAS * mcse:.4f})"
    )


def _assert_coverage(results: dict[str, np.ndarray], label: str) -> None:
    coverage = results["hits"].mean()
    assert abs(coverage - NOMINAL_COVERAGE) < COVERAGE_TOLERANCE, (
        f"{label} coverage={coverage:.3f} deviates from nominal {NOMINAL_COVERAGE} "
        f"by more than {COVERAGE_TOLERANCE}"
    )


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linear_simulation_results() -> dict[str, np.ndarray]:
    pytest.importorskip("sklearn")
    return _run_simulations(pfg.LinearModel())


@pytest.fixture(scope="module")
def xgboost_simulation_results() -> dict[str, np.ndarray]:
    pytest.importorskip("xgboost")
    return _run_simulations(pfg.XGBoostModel(tune=False))


@pytest.fixture(scope="module")
def pfn_simulation_results() -> dict[str, np.ndarray]:
    return _run_simulations(pfg.PFNModel(backend="tabicl"))


# ---------------------------------------------------------------------------
# DifferenceInMeans
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dim_simulation_results() -> dict[str, np.ndarray]:
    return _run_dim_simulations()


def test_dim_bias_and_coverage(dim_simulation_results: dict[str, np.ndarray]) -> None:
    _assert_bias(dim_simulation_results, "DifferenceInMeans")
    _assert_coverage(dim_simulation_results, "DifferenceInMeans")


# ---------------------------------------------------------------------------
# LinearModel
# ---------------------------------------------------------------------------


def test_aipw_bias_and_coverage_linear(
    linear_simulation_results: dict[str, np.ndarray],
) -> None:
    _assert_bias(linear_simulation_results, "LinearModel")
    _assert_coverage(linear_simulation_results, "LinearModel")


# ---------------------------------------------------------------------------
# XGBoostModel  (opt-in: -m ml_models)
# ---------------------------------------------------------------------------


@pytest.mark.ml_models
@pytest.mark.xgboost
def test_aipw_bias_and_coverage_xgboost(
    xgboost_simulation_results: dict[str, np.ndarray],
) -> None:
    _assert_bias(xgboost_simulation_results, "XGBoostModel")
    _assert_coverage(xgboost_simulation_results, "XGBoostModel")


# ---------------------------------------------------------------------------
# PFNModel / tabpfn  (opt-in: -m ml_models)
# ---------------------------------------------------------------------------


@pytest.mark.ml_models
@pytest.mark.pfn
def test_aipw_bias_and_coverage_pfn(
    pfn_simulation_results: dict[str, np.ndarray],
) -> None:
    _assert_bias(pfn_simulation_results, "PFNModel")
    _assert_coverage(pfn_simulation_results, "PFNModel")
