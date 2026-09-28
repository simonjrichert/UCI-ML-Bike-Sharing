"""Show what every variant predicts, for one hour and one day, side by side."""

import argparse

import numpy as np
import pandas as pd

from .data import FEATURES, features_and_target, load_hours, timestamps
from .evaluate import rmse
from .models import SHORT_LABELS, build_recipes
from .splits import TIME_SPLIT_CUTOFF, time_split

DAY_SCALE = 40  # rentals per character in the shape column

# The only variant with one coefficient per feature, so the only one whose
# prediction can be broken into a readable list of contributions.
COEFFICIENT_MODEL = "linear, hour as a number"


def choose_day(test: pd.DataFrame, rng: np.random.Generator) -> pd.Timestamp:
    """Pick one of the held-out days at random."""
    days = test["dteday"].drop_duplicates().to_numpy()
    return pd.Timestamp(days[rng.integers(len(days))])


def busiest_hour(day_rows: pd.DataFrame) -> int:
    """The hour that actually saw the most rentals, so the breakdown is interesting."""
    return int(day_rows.loc[day_rows["cnt"].idxmax(), "hr"])


def fit_all(train: pd.DataFrame) -> dict:
    """Fit every variant on the training hours, reporting progress as it goes."""
    features, target = features_and_target(train)
    fitted = {}
    for name, recipe in build_recipes().items():
        print(f"  fitting {name} ...")
        fitted[name] = recipe.fit(features, target)
    return fitted


def contribution_table(model, features_row: pd.DataFrame) -> str:
    """Every feature times its coefficient: the arithmetic behind one prediction."""
    values = features_row.iloc[0]
    rows = [
        (name, float(values[name]), coefficient, float(values[name]) * coefficient)
        for name, coefficient in zip(FEATURES, model.coef_)
    ]
    rows.sort(key=lambda item: abs(item[3]), reverse=True)

    lines = [
        f"  {'feature':<12} {'value':>8} {'coefficient':>13} {'contribution':>13}",
        f"  {'intercept':<12} {'-':>8} {'-':>13} {model.intercept_:>+13.1f}",
    ]
    lines += [
        f"  {name:<12} {value:>8.2f} {coefficient:>+13.2f} {contribution:>+13.1f}"
        for name, value, coefficient, contribution in rows
    ]
    return "\n".join(lines)


def hour_comparison(fitted: dict, features_row: pd.DataFrame, actual: float) -> str:
    """What each variant predicted for this one hour."""
    lines = [f"  {'model':<32} {'prediction':>11} {'error':>8}"]
    for name, model in fitted.items():
        prediction = float(model.predict(features_row)[0])
        lines.append(f"  {name:<32} {prediction:>11.0f} {prediction - actual:>+8.0f}")
    return "\n".join(lines)


def day_table(fitted: dict, day_rows: pd.DataFrame) -> str:
    """Every hour of the day, actual against all five variants."""
    features, target = features_and_target(day_rows)
    actual = target.to_numpy()
    predicted = {name: model.predict(features) for name, model in fitted.items()}

    columns = "".join(f" {SHORT_LABELS[name]:>9}" for name in fitted)
    header = f"  {'hr':>3} {'actual':>7}{columns}   actual shape"
    lines = [header, "  " + "-" * (len(header) - 2)]

    for position, (hour, truth) in enumerate(zip(day_rows["hr"].to_numpy(), actual)):
        row = f"  {hour:>3} {truth:>7.0f}"
        row += "".join(f" {predicted[name][position]:>9.0f}" for name in fitted)
        lines.append(f"{row}   {'#' * round(max(truth, 0) / DAY_SCALE)}")

    scores = "".join(f" {rmse(actual, predicted[name]):>9.1f}" for name in fitted)
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(f"  {'':>3} {'RMSE':>7}{scores}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", help="pin a held-out day, e.g. 2012-11-05")
    parser.add_argument("--hour", type=int, help="pin the hour used for the breakdown")
    parser.add_argument("--seed", type=int, help="reproduce a particular random day")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = time_split(load_hours())

    print(
        f"Training 5 variants on {len(split.train):,} hours "
        f"before {TIME_SPLIT_CUTOFF.date()}:"
    )
    fitted = fit_all(split.train)

    if args.day:
        day = pd.Timestamp(args.day)
    else:
        day = choose_day(split.test, np.random.default_rng(args.seed))

    day_rows = split.test[split.test["dteday"] == day]
    if day_rows.empty:
        first = split.test["dteday"].min().date()
        last = split.test["dteday"].max().date()
        raise SystemExit(f"{day.date()} is not a held-out day (pick {first} to {last})")

    hour = args.hour if args.hour is not None else busiest_hour(day_rows)
    chosen = timestamps(day_rows) == day + pd.Timedelta(hours=hour)
    if not chosen.any():
        raise SystemExit(f"hour {hour} is missing from {day.date()}")

    features_row = day_rows.loc[chosen, list(FEATURES)]
    actual = float(day_rows.loc[chosen, "cnt"].iloc[0])
    label = "Busiest hour of that day" if args.hour is None else "Chosen hour"

    print(f"\n\nHeld-out day: {day:%Y-%m-%d} (a {day:%A})")
    print(f"{label}: {hour:02d}:00, when {actual:.0f} bikes actually went out")

    print(f"\n\nHow '{COEFFICIENT_MODEL}' reaches its number for {hour:02d}:00")
    print("(the only variant with one coefficient per feature)\n")
    print(contribution_table(fitted[COEFFICIENT_MODEL], features_row))

    print(f"\n\nWhat every variant predicted for {hour:02d}:00\n")
    print(hour_comparison(fitted, features_row, actual))

    print(f"\n\nThe whole day: {day:%Y-%m-%d}")
    print(f"(one # = {DAY_SCALE} rentals; RMSE row is for this day only)\n")
    print(day_table(fitted, day_rows))

    test_features, test_target = features_and_target(split.test)
    scores = {
        name: rmse(test_target, model.predict(test_features))
        for name, model in fitted.items()
    }
    print(f"\n\nRMSE across all {len(split.test):,} held-out hours\n")
    for name, score in scores.items():
        print(f"  {name:<32} {score:>7.1f}")
    best = min(scores, key=scores.get)
    print(f"\nBest overall: {best} at {scores[best]:.1f} rentals per hour.")


if __name__ == "__main__":
    main()
