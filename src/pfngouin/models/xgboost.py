from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .base import BaseOutcomeModel

_DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
}


class XGBoostModel(BaseOutcomeModel):
    """Outcome model backed by XGBoost's XGBRegressor.

    Requires xgboost: ``pip install xgboost``.

    Parameters
    ----------
    tune:
        If True, runs a randomised hyperparameter search (via
        ``RandomizedSearchCV``) over a default grid covering
        ``n_estimators``, ``max_depth``, ``learning_rate``,
        ``subsample``, ``colsample_bytree``, and ``min_child_weight``.
    n_iter:
        Number of random parameter settings sampled when ``tune=True``.
    cv:
        Number of cross-validation folds used during tuning.
    n_jobs:
        Number of parallel jobs for ``RandomizedSearchCV`` when
        ``tune=True``.  ``-1`` uses all available CPU cores.  Default 1
        (sequential).
    param_grid:
        Override the default hyperparameter search grid.  Must be a dict
        mapping XGBRegressor parameter names to lists of candidate values.
    random_state:
        Seed for the randomised search (reproducibility).
    **xgb_kwargs:
        Passed directly to XGBRegressor (ignored for tuned parameters when
        ``tune=True``).
    """

    crossfit_required: ClassVar[bool] = True

    def __init__(
        self,
        tune: bool = True,
        n_iter: int = 250,
        cv: int = 5,
        n_jobs: int = -1,
        param_grid: dict[str, list[Any]] | None = None,
        random_state: int | None = None,
        **xgb_kwargs: object,
    ) -> None:
        try:
            from xgboost import XGBRegressor  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("XGBoostModel requires xgboost.") from exc
        self._tune = tune
        self._n_iter = n_iter
        self._cv = cv
        self._n_jobs = n_jobs
        self._param_grid = param_grid if param_grid is not None else _DEFAULT_PARAM_GRID
        self._random_state = random_state
        self._xgb_kwargs = xgb_kwargs
        self._model = XGBRegressor(**xgb_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        if self._tune:
            from sklearn.model_selection import (
                RandomizedSearchCV,  # type: ignore[import]
            )
            from xgboost import XGBRegressor  # type: ignore[import]

            base = XGBRegressor(verbosity=0, **self._xgb_kwargs)
            search = RandomizedSearchCV(
                base,
                self._param_grid,
                n_iter=self._n_iter,
                cv=self._cv,
                scoring="neg_mean_squared_error",
                random_state=self._random_state,
                refit=True,
                n_jobs=self._n_jobs,
            )
            search.fit(X, y)
            self._model = search.best_estimator_
        else:
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=float)
