import pandas as pd
import pytest

import pfngouin as pfg

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_model() -> pfg.LinearModel:
    pytest.importorskip("sklearn")
    return pfg.LinearModel()


# ---------------------------------------------------------------------------
# ttest
# ---------------------------------------------------------------------------


def test_ttest_returns_dataframe(
    ab_data: dict, linear_model: pfg.LinearModel
) -> None:  # type: ignore[type-arg]
    result = pfg.ttest(
        ab_data["a"],
        ab_data["b"],
        covariates_control=ab_data["covariates_a"],
        covariates_treatment=ab_data["covariates_b"],
        model=linear_model,
        random_state=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "p_val" in result.columns
    assert "var_reduction" in result.columns


def test_ttest_var_reduction_in_range(
    ab_data: dict, linear_model: pfg.LinearModel
) -> None:  # type: ignore[type-arg]
    result = pfg.ttest(
        ab_data["a"],
        ab_data["b"],
        covariates_control=ab_data["covariates_a"],
        covariates_treatment=ab_data["covariates_b"],
        model=linear_model,
        random_state=0,
    )
    vr = result["var_reduction"].iloc[0]
    assert 0.0 <= vr <= 1.0


def test_ttest_reduces_pvalue(ab_data: dict, linear_model: pfg.LinearModel) -> None:  # type: ignore[type-arg]
    # With a strong covariate (fixed seed), adjusted p-value should be <= original.
    import pingouin as pg

    original = pg.ttest(ab_data["b"], ab_data["a"])
    adjusted = pfg.ttest(
        ab_data["a"],
        ab_data["b"],
        covariates_control=ab_data["covariates_a"],
        covariates_treatment=ab_data["covariates_b"],
        model=linear_model,
        random_state=0,
    )
    assert adjusted["p_val"].iloc[0] <= original["p_val"].iloc[0]


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_mismatched_control_covariates_raises(
    ab_data: dict, linear_model: pfg.LinearModel
) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="control"):
        pfg.ttest(
            ab_data["a"],
            ab_data["b"],
            covariates_control=ab_data["covariates_a"][:-1],  # one row short
            covariates_treatment=ab_data["covariates_b"],
            model=linear_model,
        )


def test_mismatched_treatment_covariates_raises(
    ab_data: dict, linear_model: pfg.LinearModel
) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="treatment"):
        pfg.ttest(
            ab_data["a"],
            ab_data["b"],
            covariates_control=ab_data["covariates_a"],
            covariates_treatment=ab_data["covariates_b"][:-1],  # one row short
            model=linear_model,
        )
