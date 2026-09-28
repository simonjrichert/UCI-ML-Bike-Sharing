"""Load the hourly rides and decide what the model is allowed to look at."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "hour.csv"

TARGET = "cnt"

# casual + registered == cnt, so either column hands the model the answer.
# instant is a row counter that encodes time order. dteday is used to cut the
# time split, never as an input.
EXCLUDED_COLUMNS = ("cnt", "casual", "registered", "instant", "dteday")

FEATURES = (
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)


def load_hours(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return every hour of 2011-2012, oldest first."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python -m bike_demand.download"
        )
    hours = pd.read_csv(path, parse_dates=["dteday"])
    return hours.sort_values(["dteday", "hr"]).reset_index(drop=True)


def features_and_target(hours: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the model's inputs and the count it has to predict."""
    return hours.loc[:, list(FEATURES)], hours[TARGET]


def timestamps(hours: pd.DataFrame) -> pd.Series:
    """Combine the date column and the hour column into a plottable instant."""
    return hours["dteday"] + pd.to_timedelta(hours["hr"], unit="h")
