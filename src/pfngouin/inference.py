"""
inference.py
------------
Pingouin-mirroring statistical tests with CUPED variance reduction.
Each function mirrors the pingouin API and returns the same DataFrame format,
with an added 'var_reduction' column showing how much variance was explained
by the covariate adjustment.

Supported tests:
    ttest       - Welch / Student t-test (pg.ttest)
    more to be added: mwu, welch_anova, chi2

Usage:
    import numpy as np
    import pfngouin

    result = pfngouin.ttest(
        control, treatment,
        covariates_control=X_control,
        covariates_treatment=X_treatment,
        model=pfngouin.LinearModel(),
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg

from ._core import _adjust
from .models.base import BaseOutcomeModel
from .models.tabpfn import TabPFNModel

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    covariates_control: np.ndarray,
    covariates_treatment: np.ndarray,
    model: BaseOutcomeModel = TabPFNModel(),
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    T-test (Welch by default) with CUPED variance reduction.

    Mirrors pg.ttest(x, y, **kwargs). The 'paired' kwarg is not supported
    here because CUPED assumes independent groups.

    Parameters
    ----------
    control, treatment          : 1-D arrays of observed outcomes
    covariates_control,
    covariates_treatment        : 2-D covariate arrays (n_users × n_features)
    model                       : outcome model for variance reduction
    n_splits                    : number of cross-fitting folds (default 5,
                                  used only when model.crossfit_required is True)
    random_state                : random seed for reproducibility
    **kwargs                    : passed directly to pg.ttest

    Returns
    -------
    pandas.DataFrame identical to pg.ttest output + 'var_reduction' column
    """
    control_adj, treatment_adj, var_reduction = _adjust(
        model,
        control,
        treatment,
        covariates_control,
        covariates_treatment,
        n_splits,
        random_state,
    )
    result: pd.DataFrame = pg.ttest(treatment_adj, control_adj, **kwargs)
    result["var_reduction"] = round(var_reduction, 4)
    return result
