"""
context.py
----------
ContextStore: stores prior data in form of DataFrames for in-context learning.
"""

from __future__ import annotations

import pandas as pd


class ContextStore:
    """Registry of DataFrames for in-context learning.

    Stores full DataFrames and returns them outer-joined on demand.
    """

    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    def add(self, name: str, df: pd.DataFrame) -> None:
        """Register a prior experiment DataFrame. Stores df as-is."""
        if "_source" in df.columns:
            raise ValueError(
                f"DataFrame '{name}' already contains a '_source' column; "
                "rename it before adding to the store."
            )
        self._store[name] = df

    def get_context(self) -> pd.DataFrame:
        """Return all stored rows as an outer-joined DataFrame.

        The returned DataFrame includes a ``_source`` column with the experiment
        name so rows can be traced back to their origin. Columns present in some
        experiments but not others are filled with NaN.
        """
        if not self._store:
            return pd.DataFrame()

        out: list[pd.DataFrame] = []
        for name, df in self._store.items():
            df_tmp = df.copy()
            df_tmp["_source"] = name
            out.append(df_tmp)

        return pd.concat(out, axis=0, join="outer").reset_index(drop=True)
