import numpy as np
import pandas as pd

from pfngouin import ContextStore

# ---------------------------------------------------------------------------
# ContextStore
# ---------------------------------------------------------------------------


def test_context_store_add_and_get() -> None:
    store = ContextStore()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(50),
            "feature_b": rng.standard_normal(50),
            "outcome": rng.standard_normal(50),
        }
    )
    store.add("exp1", df)
    ctx = store.get_context()

    assert "_source" in ctx.columns
    assert len(ctx) == 50
    assert (ctx["_source"] == "exp1").all()


def test_context_store_column_alignment() -> None:
    store = ContextStore()
    rng = np.random.default_rng(1)
    df1 = pd.DataFrame(
        {
            "a": rng.standard_normal(10),
            "b": rng.standard_normal(10),
            "y": rng.standard_normal(10),
        }
    )
    df2 = pd.DataFrame(
        {
            "a": rng.standard_normal(10),
            "c": rng.standard_normal(10),
            "y": rng.standard_normal(10),
        }
    )
    store.add("exp1", df1)
    store.add("exp2", df2)
    ctx = store.get_context()

    assert len(ctx) == 20
    assert "_source" in ctx.columns

    exp1_rows = ctx[ctx["_source"] == "exp1"]
    assert exp1_rows["c"].isna().all(), "exp1 rows should have NaN for col 'c'"

    exp2_rows = ctx[ctx["_source"] == "exp2"]
    assert exp2_rows["b"].isna().all(), "exp2 rows should have NaN for col 'b'"


def test_context_store_empty() -> None:
    store = ContextStore()
    ctx = store.get_context()
    assert ctx.empty
