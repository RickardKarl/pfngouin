import numpy as np
import pandas as pd
import pytest

import pfngouin as pfg


@pytest.fixture()
def linear_model() -> pfg.LinearModel:
    pytest.importorskip("sklearn")
    return pfg.LinearModel()


@pytest.fixture()
def xgboost_model() -> pfg.XGBoostModel:
    pytest.importorskip("xgboost")
    return pfg.XGBoostModel(tune=False)


@pytest.fixture()
def tabpfn_model() -> pfg.PFNModel:
    pytest.importorskip("tabpfn_client")
    return pfg.PFNModel()


@pytest.fixture()
def ab_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    pre_revenue = rng.normal(10, 3, n * 2)
    noise = rng.normal(0, 1, n * 2)
    outcome = pre_revenue * 0.9 + noise
    outcome[n:] += 2.0
    return pd.DataFrame({
        "outcome": outcome,
        "group": ["control"] * n + ["treatment"] * n,
        "pre_revenue": pre_revenue,
    })


@pytest.fixture()
def multi_group_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    pre_revenue = rng.normal(10, 3, n * 3)
    noise = rng.normal(0, 1, n * 3)
    outcome = pre_revenue * 0.9 + noise
    outcome[n : 2 * n] += 1.0   # group B effect
    outcome[2 * n :] += 2.5     # group C effect
    return pd.DataFrame({
        "revenue": outcome,
        "group": ["A"] * n + ["B"] * n + ["C"] * n,
        "pre_revenue": pre_revenue,
    })
