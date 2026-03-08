import numpy as np
import pytest


@pytest.fixture()
def ab_data() -> dict:  # type: ignore[type-arg]
    rng = np.random.default_rng(42)
    n = 100
    pre_revenue = rng.normal(10, 3, n * 2)
    noise = rng.normal(0, 1, n * 2)
    outcome = pre_revenue * 0.9 + noise
    outcome[n:] += 2.0
    return {
        "a": outcome[:n],
        "b": outcome[n:],
        "covariates_a": pre_revenue[:n].reshape(-1, 1),
        "covariates_b": pre_revenue[n:].reshape(-1, 1),
    }
