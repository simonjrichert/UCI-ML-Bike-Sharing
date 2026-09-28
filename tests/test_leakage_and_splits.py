"""Guard the two things that decide whether the comparison means anything."""

import numpy as np
import pandas as pd
import pytest

from bike_demand.data import EXCLUDED_COLUMNS, FEATURES, TARGET, features_and_target
from bike_demand.evaluate import rmse
from bike_demand.splits import TIME_SPLIT_CUTOFF, random_split, time_split
from bike_demand.train import BASELINE, fit_models


@pytest.fixture
def hours() -> pd.DataFrame:
    """Two years of synthetic hours with the same columns as hour.csv."""
    rows = 24 * 730
    dates = pd.Timestamp("2011-01-01") + pd.to_timedelta(np.arange(rows) // 24, unit="D")
    hour_of_day = np.arange(rows) % 24
    rng = np.random.default_rng(0)
    casual = rng.integers(0, 50, rows)
    registered = rng.integers(0, 300, rows)
    frame = pd.DataFrame(
        {
            "instant": np.arange(1, rows + 1),
            "dteday": dates,
            "hr": hour_of_day,
            "casual": casual,
            "registered": registered,
            "cnt": casual + registered,
        }
    )
    for column in FEATURES:
        if column not in frame:
            frame[column] = rng.random(rows)
    return frame


def test_features_exclude_the_answer(hours):
    features, target = features_and_target(hours)
    leaked = set(features.columns) & set(EXCLUDED_COLUMNS)
    assert not leaked, f"these columns hand the model the answer: {leaked}"
    assert target.name == TARGET


def test_casual_plus_registered_is_the_target(hours):
    # The reason casual and registered are excluded, asserted rather than assumed.
    assert (hours["casual"] + hours["registered"] == hours[TARGET]).all()


def test_time_split_never_trains_on_the_test_period(hours):
    split = time_split(hours)
    assert split.train["dteday"].max() < TIME_SPLIT_CUTOFF <= split.test["dteday"].min()
    assert not set(split.train["dteday"]) & set(split.test["dteday"])
    assert len(split.train) + len(split.test) == len(hours)


def test_random_split_mixes_the_same_days_into_both_sides(hours):
    # Not a bug in random_split: this overlap is the effect the project measures.
    split = random_split(hours)
    assert set(split.train["dteday"]) & set(split.test["dteday"])
    assert not set(split.train["instant"]) & set(split.test["instant"])
    assert len(split.train) + len(split.test) == len(hours)


def test_baseline_predicts_the_training_mean(hours):
    split = time_split(hours)
    baseline = fit_models(split.train)[BASELINE]
    test_features, _ = features_and_target(split.test)
    predictions = baseline.predict(test_features)
    assert len(predictions) == len(split.test)
    assert np.allclose(predictions, split.train[TARGET].mean())


def test_rmse_is_zero_for_perfect_predictions():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(3.5355, abs=1e-4)
