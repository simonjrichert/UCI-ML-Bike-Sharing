"""Fit the two predictors: a mean guess and a linear model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .data import features_and_target

BASELINE = "mean baseline"
MODEL = "linear regression"


class MeanBaseline:
    """Predicts the training-set average rental count for every hour."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "MeanBaseline":
        self.mean_ = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean_)


def fit_models(train: pd.DataFrame) -> dict:
    """Fit both predictors on the training rows only."""
    features, target = features_and_target(train)
    return {
        BASELINE: MeanBaseline().fit(features, target),
        MODEL: LinearRegression().fit(features, target),
    }
