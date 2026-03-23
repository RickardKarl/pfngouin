"""
hypothesis_testing.py
---------------------
Statistical tests with CUPED variance reduction.
Each function accepts a DataFrame with outcome, group, and covariate columns,
and returns a results DataFrame with a 'var_reduction' column showing how much
variance was explained by the covariate adjustment.

Supported tests:
    ttest       - Welch / Student t-test
    mwu         - Mann-Whitney U test
    tost        - Two One-Sided Test for equivalence
    anova       - One-way ANOVA
    welch_anova - Welch's ANOVA
    kruskal     - Kruskal-Wallis H-test

Usage:
    import pfngouin

    result = pfngouin.ttest(
        data,
        dv="revenue",
        between="group",
        covar=["pre1", "pre2", "pre3"],
        model=pfngouin.LinearModel(),
    )
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pingouin as pg

from ._core import _adjust_df
from .models.base import _AnyModel

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _default_model() -> _AnyModel:
    from .models.pfn import PFNModel

    return PFNModel()


def _binary_adjust(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    model: _AnyModel | None,
    n_splits: int,
    random_state: int | None,
    caller: str,
) -> tuple[pd.Series, pd.Series, float]:
    """Validate binary group, apply CUPED, return (ctrl_adj, trt_adj, var_reduction)."""
    groups = sorted(data[between].unique())
    if len(groups) != 2:
        raise ValueError(
            f"{caller} requires exactly 2 groups in '{between}', "
            f"found {len(groups)}: {groups}"
        )
    if model is None:
        model = _default_model()
    adjusted_data, var_reduction = _adjust_df(
        data, dv, covar, model, n_splits, random_state
    )
    ctrl_adj: pd.Series = adjusted_data.loc[data[between] == groups[0], dv]
    trt_adj: pd.Series = adjusted_data.loc[data[between] == groups[1], dv]
    return ctrl_adj, trt_adj, var_reduction


def _multigroup_adjust(
    data: pd.DataFrame,
    dv: str,
    covar: str | list[str],
    model: _AnyModel | None,
    n_splits: int,
    random_state: int | None,
) -> tuple[pd.DataFrame, float]:
    """Apply CUPED, return (adjusted_data, var_reduction)."""
    if model is None:
        model = _default_model()
    return _adjust_df(data, dv, covar, model, n_splits, random_state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ttest(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    model: _AnyModel | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    T-test (Welch by default) with CUPED variance reduction.

    The 'paired' kwarg is not supported because CUPED assumes independent groups.

    Parameters
    ----------
    data         : DataFrame containing outcome, group, and covariate columns
    dv           : name of the dependent variable (outcome) column
    between      : name of the binary group column (exactly 2 unique values).
                   Groups are sorted; the first is treated as control, the
                   second as treatment.
    covar        : covariate column name(s) for CUPED adjustment
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (default 5,
                   used only when model.crossfit_required is True)
    random_state : random seed for reproducibility
    **kwargs     : additional keyword arguments forwarded to the underlying test

    Returns
    -------
    pandas.DataFrame with test results and a 'var_reduction' column
    """
    ctrl_adj, trt_adj, var_reduction = _binary_adjust(
        data, dv, between, covar, model, n_splits, random_state, "ttest"
    )
    result: pd.DataFrame = pg.ttest(trt_adj, ctrl_adj, **kwargs)
    result["var_reduction"] = round(var_reduction, 4)
    return result


def mwu(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    model: _AnyModel | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Mann-Whitney U test with CUPED variance reduction.

    Parameters
    ----------
    data         : DataFrame containing outcome, group, and covariate columns
    dv           : name of the dependent variable (outcome) column
    between      : name of the binary group column (exactly 2 unique values).
                   Groups are sorted; the first is treated as control, the
                   second as treatment.
    covar        : covariate column name(s) for CUPED adjustment
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (default 5,
                   used only when model.crossfit_required is True)
    random_state : random seed for reproducibility
    **kwargs     : additional keyword arguments forwarded to the underlying test

    Returns
    -------
    pandas.DataFrame with test results and a 'var_reduction' column
    """
    ctrl_adj, trt_adj, var_reduction = _binary_adjust(
        data, dv, between, covar, model, n_splits, random_state, "mwu"
    )
    result: pd.DataFrame = pg.mwu(trt_adj, ctrl_adj, **kwargs)
    result["var_reduction"] = round(var_reduction, 4)
    return result


def tost(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    bound: float = 1.0,
    model: _AnyModel | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Two One-Sided Test (TOST) for equivalence with CUPED variance reduction.

    Used for equivalence and non-inferiority testing. The 'paired' kwarg is
    not supported because CUPED assumes independent groups.

    Note: the p-value column in the output is 'pval' (pingouin naming),
    not 'p_val' as in ttest/mwu.

    Parameters
    ----------
    data         : DataFrame containing outcome, group, and covariate columns
    dv           : name of the dependent variable (outcome) column
    between      : name of the binary group column (exactly 2 unique values).
                   Groups are sorted; the first is treated as control, the
                   second as treatment.
    covar        : covariate column name(s) for CUPED adjustment
    bound        : equivalence bound (passed to pg.tost)
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (default 5,
                   used only when model.crossfit_required is True)
    random_state : random seed for reproducibility
    **kwargs     : additional keyword arguments forwarded to the underlying test

    Returns
    -------
    pandas.DataFrame with test results and a 'var_reduction' column
    """
    ctrl_adj, trt_adj, var_reduction = _binary_adjust(
        data, dv, between, covar, model, n_splits, random_state, "tost"
    )
    result: pd.DataFrame = pg.tost(trt_adj, ctrl_adj, bound=bound, **kwargs)
    result["var_reduction"] = round(var_reduction, 4)
    return result


def anova(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    model: _AnyModel | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    One-way ANOVA with CUPED variance reduction.

    Suitable for multi-arm experiments with 3+ groups.

    Parameters
    ----------
    data         : DataFrame containing outcome, group, and covariate columns
    dv           : name of the dependent variable (outcome) column
    between      : name of the between-subject factor (group) column
    covar        : covariate column name(s) for CUPED adjustment
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (default 5,
                   used only when model.crossfit_required is True)
    random_state : random seed for reproducibility
    **kwargs     : additional keyword arguments forwarded to the underlying test

    Returns
    -------
    pandas.DataFrame with test results and a 'var_reduction' column.
    p-value column is 'p_unc'
    """
    adjusted_data, var_reduction = _multigroup_adjust(
        data, dv, covar, model, n_splits, random_state
    )
    result: pd.DataFrame = pg.anova(
        data=adjusted_data, dv=dv, between=between, **kwargs
    )
    result["var_reduction"] = round(var_reduction, 4)
    return result


def welch_anova(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    model: _AnyModel | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Welch's ANOVA with CUPED variance reduction.

    Does not assume equal variances across groups. Suitable for multi-arm experiments.

    Parameters
    ----------
    data         : DataFrame containing outcome, group, and covariate columns
    dv           : name of the dependent variable (outcome) column
    between      : name of the between-subject factor (group) column
    covar        : covariate column name(s) for CUPED adjustment
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (default 5,
                   used only when model.crossfit_required is True)
    random_state : random seed for reproducibility
    **kwargs     : additional keyword arguments forwarded to the underlying test

    Returns
    -------
    pandas.DataFrame with test results and a 'var_reduction' column.
    p-value column is 'p_unc'
    """
    adjusted_data, var_reduction = _multigroup_adjust(
        data, dv, covar, model, n_splits, random_state
    )
    result: pd.DataFrame = pg.welch_anova(
        data=adjusted_data, dv=dv, between=between, **kwargs
    )
    result["var_reduction"] = round(var_reduction, 4)
    return result


def kruskal(
    data: pd.DataFrame,
    dv: str,
    between: str,
    covar: str | list[str],
    model: _AnyModel | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test with CUPED variance reduction.

    Non-parametric alternative to one-way ANOVA for multi-arm experiments.

    Parameters
    ----------
    data         : DataFrame containing outcome, group, and covariate columns
    dv           : name of the dependent variable (outcome) column
    between      : name of the between-subject factor (group) column
    covar        : covariate column name(s) for CUPED adjustment
    model        : outcome model for variance reduction
    n_splits     : number of cross-fitting folds (default 5,
                   used only when model.crossfit_required is True)
    random_state : random seed for reproducibility
    **kwargs     : additional keyword arguments forwarded to the underlying test

    Returns
    -------
    pandas.DataFrame with test results and a 'var_reduction' column.
    p-value column is 'p_unc'
    """
    adjusted_data, var_reduction = _multigroup_adjust(
        data, dv, covar, model, n_splits, random_state
    )
    result: pd.DataFrame = pg.kruskal(
        data=adjusted_data, dv=dv, between=between, **kwargs
    )
    result["var_reduction"] = round(var_reduction, 4)
    return result
