"""Check that every ablation variant trains and that the interaction encoding is right."""

import numpy as np
import pandas as pd
import pytest

from bike_demand.data import FEATURES, TARGET, features_and_target
from bike_demand.models import _hour_by_workingday, build_recipes


@pytest.fixture
def hours() -> pd.DataFrame:
    """Synthetic hours with realistic dtypes for `hr` and `workingday`."""
    rows = 24 * 60
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "hr": np.arange(rows) % 24,
            "workingday": rng.integers(0, 2, rows),
            TARGET: rng.integers(0, 700, rows),
        }
    )
    for column in FEATURES:
        if column not in frame:
            frame[column] = rng.random(rows)
    return frame


@pytest.mark.parametrize("name", list(build_recipes()))
def test_every_recipe_fits_and_predicts_one_value_per_row(name, hours):
    features, target = features_and_target(hours)
    recipe = build_recipes()[name]
    recipe.fit(features, target)
    predictions = recipe.predict(features)
    assert np.shape(predictions) == (len(hours),)
    assert np.isfinite(predictions).all()


def test_hour_by_workingday_gives_each_hour_two_categories():
    frame = pd.DataFrame(
        {
            "hr": list(range(24)) * 2,
            "workingday": [0] * 24 + [1] * 24,
        }
    )
    combined = _hour_by_workingday(frame)
    assert combined.shape == (48, 1)
    assert len(np.unique(combined)) == 48

    # 8am on a day off and 8am on a working day must not share a coefficient.
    day_off = _hour_by_workingday(pd.DataFrame({"hr": [8], "workingday": [0]}))
    working = _hour_by_workingday(pd.DataFrame({"hr": [8], "workingday": [1]}))
    assert day_off.item() == 8
    assert working.item() == 32
