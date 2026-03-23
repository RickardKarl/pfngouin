"""
datasets.py
-----------
Synthetic A/B test data for notebooks and tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_experiment_data(
    N: int = 300,
    p_treatment: float = 0.5,
    effect: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
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
    DataFrame with columns: outcome, group ("control"/"treatment"), pre1, pre2, pre3
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

    # Apply treatment effect
    outcome[in_treatment] += effect

    group = np.where(in_treatment, "treatment", "control")
    return pd.DataFrame({"outcome": outcome, "group": group, "pre1": pre1, "pre2": pre2, "pre3": pre3})


def make_multigroup_df(
    N: int,
    effects: tuple[float, ...],
    seed: int,
) -> pd.DataFrame:
    """Multi-group experiment data; mirrors make_experiment_data's latent structure."""
    rng = np.random.default_rng(seed)
    n_per_group = N // len(effects)
    rows = []
    for i, effect in enumerate(effects):
        latent = rng.normal(10, 5, n_per_group)
        pre1 = np.sin(latent / 3.0) * latent + rng.normal(0, 2, n_per_group)
        pre2 = 0.1 * latent**3 - 2.0 * latent + rng.normal(0, 3, n_per_group)
        pre3 = np.exp(0.1 * latent) + rng.normal(0, 4, n_per_group)
        noise = rng.normal(0, 8, n_per_group)
        outcome = np.tanh(latent / 5.0) * 10 + 0.05 * latent**2 + noise + effect
        rows.append(
            pd.DataFrame(
                {
                    "outcome": outcome,
                    "group": f"G{i}",
                    "pre1": pre1,
                    "pre2": pre2,
                    "pre3": pre3,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
