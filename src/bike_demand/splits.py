"""The two ways of dividing the same hours into training and test rows."""

from dataclasses import dataclass

import pandas as pd

TIME_SPLIT_CUTOFF = pd.Timestamp("2012-11-01")
TEST_FRACTION = 0.2
RANDOM_SEED = 0


@dataclass(frozen=True)
class Split:
    name: str
    description: str
    train: pd.DataFrame
    test: pd.DataFrame


def random_split(
    hours: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> Split:
    """Shuffle the hours and cut, ignoring when each one happened."""
    shuffled = hours.sample(frac=1.0, random_state=seed)
    test_rows = int(len(shuffled) * test_fraction)
    return Split(
        name="random split",
        description=f"{int(test_fraction * 100)}% of hours held out, drawn from anywhere in 2011-2012",
        train=shuffled.iloc[test_rows:].sort_index().copy(),
        test=shuffled.iloc[:test_rows].sort_index().copy(),
    )


def time_split(hours: pd.DataFrame, cutoff: pd.Timestamp = TIME_SPLIT_CUTOFF) -> Split:
    """Train on everything before the cutoff, test on everything after it."""
    return Split(
        name="time split",
        description=f"trained on hours before {cutoff.date()}, tested on the hours after",
        train=hours[hours["dteday"] < cutoff].copy(),
        test=hours[hours["dteday"] >= cutoff].copy(),
    )
