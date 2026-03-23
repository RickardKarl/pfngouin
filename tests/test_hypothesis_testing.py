import pandas as pd
import pytest

import pfngouin as pfg

# ---------------------------------------------------------------------------
# ttest
# ---------------------------------------------------------------------------


def test_ttest_returns_dataframe(ab_data: pd.DataFrame, linear_model: pfg.LinearModel) -> None:
    result = pfg.ttest(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "p_val" in result.columns
    assert "var_reduction" in result.columns


def test_ttest_var_reduction_in_range(
    ab_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.ttest(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_ttest_reduces_pvalue(ab_data: pd.DataFrame, linear_model: pfg.LinearModel) -> None:
    import pingouin as pg

    # With a strong covariate (fixed seed), adjusted p-value should be <= original.
    trt = ab_data.loc[ab_data["group"] == "treatment", "outcome"]
    ctrl = ab_data.loc[ab_data["group"] == "control", "outcome"]
    original = pg.ttest(trt, ctrl)
    adjusted = pfg.ttest(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert adjusted["p_val"].iloc[0] <= original["p_val"].iloc[0]


# ---------------------------------------------------------------------------
# mwu
# ---------------------------------------------------------------------------


def test_mwu_returns_dataframe(ab_data: pd.DataFrame, linear_model: pfg.LinearModel) -> None:
    result = pfg.mwu(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "p_val" in result.columns
    assert "var_reduction" in result.columns


def test_mwu_var_reduction_in_range(
    ab_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.mwu(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_mwu_reduces_pvalue(ab_data: pd.DataFrame, linear_model: pfg.LinearModel) -> None:
    import pingouin as pg

    trt = ab_data.loc[ab_data["group"] == "treatment", "outcome"]
    ctrl = ab_data.loc[ab_data["group"] == "control", "outcome"]
    original = pg.mwu(trt, ctrl)
    adjusted = pfg.mwu(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert adjusted["p_val"].iloc[0] <= original["p_val"].iloc[0]


# ---------------------------------------------------------------------------
# tost
# ---------------------------------------------------------------------------


def test_tost_returns_dataframe(ab_data: pd.DataFrame, linear_model: pfg.LinearModel) -> None:
    result = pfg.tost(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    # pingouin.tost uses 'pval', not 'p_val'
    assert "pval" in result.columns
    assert "var_reduction" in result.columns


def test_tost_var_reduction_in_range(
    ab_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.tost(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_tost_reduces_pvalue(ab_data: pd.DataFrame, linear_model: pfg.LinearModel) -> None:
    import pingouin as pg

    # ab_data has treatment effect=2.0; bound=3.0 includes this, so tost
    # can confirm equivalence. With a strong covariate, CUPED should give
    # a smaller (more significant) pval.
    trt = ab_data.loc[ab_data["group"] == "treatment", "outcome"]
    ctrl = ab_data.loc[ab_data["group"] == "control", "outcome"]
    original = pg.tost(trt, ctrl, bound=3.0)
    adjusted = pfg.tost(
        ab_data,
        dv="outcome",
        between="group",
        covar="pre_revenue",
        bound=3.0,
        model=linear_model,
        random_state=0,
    )
    assert adjusted["pval"].iloc[0] <= original["pval"].iloc[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_ttest_non_binary_between_raises(
    ab_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    data = ab_data.copy()
    data.loc[data.index[:10], "group"] = "other"
    with pytest.raises(ValueError, match="exactly 2 groups"):
        pfg.ttest(data, dv="outcome", between="group", covar="pre_revenue", model=linear_model)


# ---------------------------------------------------------------------------
# anova
# ---------------------------------------------------------------------------


def test_anova_returns_dataframe(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "p_unc" in result.columns
    assert "var_reduction" in result.columns


def test_anova_var_reduction_in_range(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_anova_reduces_pvalue(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    import pingouin as pg

    original = pg.anova(data=multi_group_data, dv="revenue", between="group")
    adjusted = pfg.anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert adjusted["p_unc"].iloc[0] <= original["p_unc"].iloc[0]


def test_anova_string_covar_works(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    pfg.anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",  # string, not list
        model=linear_model,
        random_state=0,
    )


def test_anova_list_covar_works(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    pfg.anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar=["pre_revenue"],  # list
        model=linear_model,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# welch_anova
# ---------------------------------------------------------------------------


def test_welch_anova_returns_dataframe(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.welch_anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "p_unc" in result.columns
    assert "var_reduction" in result.columns


def test_welch_anova_var_reduction_in_range(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.welch_anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_welch_anova_reduces_pvalue(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    import pingouin as pg

    original = pg.welch_anova(data=multi_group_data, dv="revenue", between="group")
    adjusted = pfg.welch_anova(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert adjusted["p_unc"].iloc[0] <= original["p_unc"].iloc[0]


# ---------------------------------------------------------------------------
# kruskal
# ---------------------------------------------------------------------------


def test_kruskal_returns_dataframe(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.kruskal(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "p_unc" in result.columns
    assert "var_reduction" in result.columns


def test_kruskal_var_reduction_in_range(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    result = pfg.kruskal(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_kruskal_reduces_pvalue(
    multi_group_data: pd.DataFrame, linear_model: pfg.LinearModel
) -> None:
    import pingouin as pg

    original = pg.kruskal(data=multi_group_data, dv="revenue", between="group")
    adjusted = pfg.kruskal(
        data=multi_group_data,
        dv="revenue",
        between="group",
        covar="pre_revenue",
        model=linear_model,
        random_state=0,
    )
    assert adjusted["p_unc"].iloc[0] <= original["p_unc"].iloc[0]
