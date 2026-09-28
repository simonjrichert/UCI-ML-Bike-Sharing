"""Model variants for the ablation. Each one changes exactly one thing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .train import MeanBaseline

HOURS_PER_DAY = 24

# Column headings for side-by-side output, where the full names do not fit.
SHORT_LABELS = {
    "mean baseline": "average",
    "linear, hour as a number": "hr-number",
    "linear, hour as 24 categories": "hr-24cat",
    "linear, hour x workingday": "hr*work",
    "gradient boosted trees": "trees",
}


def _hour_by_workingday(frame: pd.DataFrame) -> np.ndarray:
    """One category per (hour, working or not) pair, so a spike can be weekday-only."""
    hour = frame["hr"].to_numpy()
    working = frame["workingday"].to_numpy()
    return (hour + HOURS_PER_DAY * working).reshape(-1, 1)


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_recipes() -> dict:
    """Fresh, unfitted variants, in the order they belong in the table."""
    return {
        "mean baseline": MeanBaseline(),
        "linear, hour as a number": LinearRegression(),
        "linear, hour as 24 categories": Pipeline(
            [
                (
                    "encode",
                    ColumnTransformer(
                        [("hour", _one_hot(), ["hr"])], remainder="passthrough"
                    ),
                ),
                ("fit", LinearRegression()),
            ]
        ),
        "linear, hour x workingday": Pipeline(
            [
                (
                    "encode",
                    ColumnTransformer(
                        [
                            (
                                "hour_working",
                                Pipeline(
                                    [
                                        ("combine", FunctionTransformer(_hour_by_workingday)),
                                        ("onehot", _one_hot()),
                                    ]
                                ),
                                ["hr", "workingday"],
                            )
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("fit", LinearRegression()),
            ]
        ),
        # Trees split on `hr < 6` by themselves, so they need none of the encoding above.
        "gradient boosted trees": HistGradientBoostingRegressor(random_state=0),
    }
