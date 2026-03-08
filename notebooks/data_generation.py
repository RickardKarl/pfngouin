"""
data_generation.py
------------------
Synthetic A/B test data for the pfngouin tutorial notebook.
"""

from __future__ import annotations

import numpy as np


def make_experiment_data(
    N: int = 300,
    p_treatment: float = 0.5,
    effect: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Simulate a randomized A/B test with a strong pre-experiment covariate.

    Parameters
    ----------
    N           : total number of users
    p_treatment : probability of being assigned to treatment (default 0.5)
    effect      : true treatment effect (additive)
    seed        : random seed for reproducibility

    Returns
    -------
    dict with keys:
        control    : 1-D outcome array for control group
        treatment  : 1-D outcome array for treatment group
        X_ctrl     : covariate matrix for control  (n_ctrl, 3)
        X_trt      : covariate matrix for treatment (n_trt, 3)
    """
    rng = np.random.default_rng(seed)

    # Shared latent user-level signal drives all covariates and the outcome
    latent = rng.normal(10, 5, N)

    # Three covariates: each a strongly nonlinear function of latent + noise
    pre1 = np.sin(latent / 3.0) * latent + rng.normal(0, 2, N)  # sinusoidal modulation
    pre2 = 0.1 * latent**3 - 2.0 * latent + rng.normal(0, 3, N)  # cubic
    pre3 = np.exp(0.1 * latent) + rng.normal(0, 4, N)  # exponential

    # Outcome: strongly nonlinear function of latent + heavy noise (low R²)
    noise = rng.normal(0, 8, N)
    outcome = np.tanh(latent / 5.0) * 10 + 0.05 * latent**2 + noise

    # Randomized group assignment (Bernoulli)
    in_treatment = rng.random(N) < p_treatment
    in_control = ~in_treatment

    # Apply treatment effect
    outcome[in_treatment] += effect

    # Covariate matrix: [pre1, pre2, pre3]
    X = np.column_stack([pre1, pre2, pre3])

    return {
        "control": outcome[in_control],
        "treatment": outcome[in_treatment],
        "X_ctrl": X[in_control],
        "X_trt": X[in_treatment],
    }
