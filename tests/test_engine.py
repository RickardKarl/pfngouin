from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfngouin import ATEResult, ContextStore, InferenceEngine, aipw
from pfngouin.models import LinearModel
from _data import make_experiment_data


def _prepare(N: int = 200, effect: float = 1.0, seed: int = 42) -> pd.DataFrame:
    df = make_experiment_data(N=N, effect=effect, seed=seed)
    return df.assign(treated=(df["group"] == "treatment").astype(float))


_COVARIATES = ["pre1", "pre2", "pre3"]


# ---------------------------------------------------------------------------
# Helper model
# ---------------------------------------------------------------------------


class _RecordingModel(LinearModel):
    """LinearModel that records the number of training samples seen."""

    n_train_samples: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _RecordingModel:
        self.n_train_samples = len(y)
        super().fit(X, y)
        return self


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


def test_inference_engine_augments_data() -> None:
    rng = np.random.default_rng(0)
    n_exp = 50
    n_ctx = 30

    X = rng.standard_normal((n_exp, 2))
    y = rng.standard_normal(n_exp)

    ctx_df = pd.DataFrame(
        {
            "f0": rng.standard_normal(n_ctx),
            "f1": rng.standard_normal(n_ctx),
            "outcome": rng.standard_normal(n_ctx),
        }
    )
    store = ContextStore()
    store.add("prior", ctx_df)

    engine = InferenceEngine(_RecordingModel(), context_store=store)
    engine.fit(pd.DataFrame(X, columns=["f0", "f1"]), pd.Series(y, name="outcome"))

    assert engine._fitted_model is not None
    fitted: _RecordingModel = engine._fitted_model  # type: ignore[assignment]
    assert fitted.n_train_samples == n_exp + n_ctx


def test_inference_engine_no_context_passthrough() -> None:
    """Empty context store → warning emitted, no augmentation."""
    rng = np.random.default_rng(0)
    n = 40
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal(n)

    engine = InferenceEngine(_RecordingModel(), context_store=ContextStore())
    with pytest.warns(UserWarning, match="outcome"):
        engine.fit(pd.DataFrame(X, columns=["f0", "f1"]), pd.Series(y, name="outcome"))

    assert engine._fitted_model is not None
    fitted: _RecordingModel = engine._fitted_model  # type: ignore[assignment]
    assert fitted.n_train_samples == n


def test_inference_engine_predict_before_fit_raises() -> None:
    engine = InferenceEngine(LinearModel(), context_store=ContextStore())
    with pytest.raises(RuntimeError, match="fitted"):
        engine.predict(np.zeros((5, 2)))


def test_inference_engine_crossfit_required_delegates() -> None:
    engine = InferenceEngine(LinearModel(), context_store=ContextStore())
    assert engine.crossfit_required == LinearModel.crossfit_required


def test_inference_engine_skips_context_without_outcome_col() -> None:
    """Context rows for a different outcome are not used; a warning is emitted."""
    rng = np.random.default_rng(2)
    n_exp, n_ctx = 20, 15

    ctx_df = pd.DataFrame(
        {
            "f0": rng.standard_normal(n_ctx),
            "other_outcome": rng.standard_normal(n_ctx),  # different outcome
        }
    )
    store = ContextStore()
    store.add("prior", ctx_df)

    engine = InferenceEngine(_RecordingModel(), context_store=store)
    with pytest.warns(UserWarning, match="outcome"):
        engine.fit(
            pd.DataFrame(rng.standard_normal((n_exp, 1)), columns=["f0"]),
            pd.Series(rng.standard_normal(n_exp), name="outcome"),
        )

    fitted: _RecordingModel = engine._fitted_model  # type: ignore[assignment]
    assert fitted.n_train_samples == n_exp  # no augmentation


# ---------------------------------------------------------------------------
# Cross-experiment context integration
# ---------------------------------------------------------------------------


def test_aipw_with_cross_experiment_context() -> None:
    """Engine uses context rows sharing the outcome; unrelated experiments are skipped.

    Note: NaN-filled heterogeneous context (different feature sets across experiments)
    requires a NaN-capable model such as TabPFN or TabICL.  This test keeps all
    feature columns consistent across experiments so LinearModel can be used.
    """
    rng = np.random.default_rng(42)
    n, n_ctx = 60, 40

    # Prior experiment: same outcome + same features as target (no NaN augmentation)
    df_prior = pd.DataFrame(
        {
            "outcome1": rng.standard_normal(n_ctx),
            "feature3": rng.standard_normal(n_ctx),
            "feature2": rng.integers(0, 2, n_ctx).astype(float),
        }
    )
    # Unrelated experiment: different outcome → should be skipped by engine
    df_other = pd.DataFrame(
        {
            "outcome2": rng.standard_normal(n_ctx),
            "feature2": rng.integers(0, 2, n_ctx).astype(float),
            "feature3": rng.standard_normal(n_ctx),
        }
    )

    context = ContextStore()
    context.add("exp_relevant", df_prior)
    context.add("exp_irrelevant", df_other)

    engine = InferenceEngine(LinearModel(), context_store=context)

    df_target = pd.DataFrame(
        {
            "outcome1": rng.standard_normal(n),
            "feature2": rng.integers(0, 2, n).astype(float),
            "feature3": rng.standard_normal(n),
        }
    )
    result = aipw(
        df_target,
        outcome="outcome1",
        treatment="feature2",
        covariates="feature3",
        model=engine,
    )

    assert isinstance(result, ATEResult)
    assert 0.0 <= result.variance_reduction <= 1.0
    assert result.ci[0] < result.ci[1]


# ---------------------------------------------------------------------------
# TabICLModel (ml_models marker — skipped unless tabicl is installed)
# ---------------------------------------------------------------------------


@pytest.mark.ml_models
@pytest.mark.tabicl
def test_tabicl_model_interface() -> None:
    pytest.importorskip("tabicl")
    import pfngouin as pfg

    df = _prepare(N=100)
    X = df[["pre1", "pre2", "pre3", "treated"]].to_numpy(dtype=float)
    Y = df["outcome"].to_numpy(dtype=float)
    model = pfg.PFNModel(backend="tabicl")
    model.fit(X, Y)
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()
