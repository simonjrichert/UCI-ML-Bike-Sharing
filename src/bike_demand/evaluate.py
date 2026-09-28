"""Score predictions and report both splits in the same units."""

import numpy as np
import pandas as pd

from .data import features_and_target
from .splits import Split
from .train import BASELINE, MODEL, fit_models


def rmse(actual, predicted) -> float:
    """Average miss in rentals per hour, with big misses weighted heavily."""
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(error**2)))


def score_split(split: Split) -> dict:
    """Fit on the training rows, then score both sets of rows."""
    models = fit_models(split.train)
    train_features, train_target = features_and_target(split.train)
    test_features, test_target = features_and_target(split.test)
    return {
        "split": split,
        "models": models,
        "train_rmse": {
            name: rmse(train_target, model.predict(train_features))
            for name, model in models.items()
        },
        "test_rmse": {
            name: rmse(test_target, model.predict(test_features))
            for name, model in models.items()
        },
    }


def _span(frame: pd.DataFrame) -> str:
    return f"{frame['dteday'].min().date()} to {frame['dteday'].max().date()}"


def report(result: dict) -> str:
    split = result["split"]
    lines = [
        f"{split.name}  ({split.description})",
        f"  train {len(split.train):>6,} hours   {_span(split.train)}",
        f"  test  {len(split.test):>6,} hours   {_span(split.test)}",
    ]
    for name in (BASELINE, MODEL):
        lines.append(
            f"  {name:<18} test RMSE {result['test_rmse'][name]:7.1f}"
            f"   train RMSE {result['train_rmse'][name]:7.1f}"
        )
    improvement = 1 - result["test_rmse"][MODEL] / result["test_rmse"][BASELINE]
    lines.append(f"  model beats the baseline by {improvement:.0%}")
    return "\n".join(lines)
