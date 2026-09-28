"""Score every model variant on the honest split, one change at a time."""

import numpy as np

from .data import features_and_target, load_hours
from .evaluate import rmse
from .models import build_recipes
from .splits import TIME_SPLIT_CUTOFF, time_split

NIGHT_HOURS = (0, 1, 2, 3, 4, 5)
RUSH_HOURS = (7, 8, 17, 18)


def main() -> None:
    split = time_split(load_hours())
    train_features, train_target = features_and_target(split.train)
    test_features, test_target = features_and_target(split.test)

    hours = split.test["hr"].to_numpy()
    night = np.isin(hours, NIGHT_HOURS)
    rush = np.isin(hours, RUSH_HOURS)

    print(
        f"Trained on {len(split.train):,} hours before {TIME_SPLIT_CUTOFF.date()}, "
        f"scored on the {len(split.test):,} hours after.\n"
        f"night = {NIGHT_HOURS[0]}h-{NIGHT_HOURS[-1]}h, rush = {RUSH_HOURS}, "
        f"neg = predictions below zero rentals.\n"
    )

    header = (
        f"  {'model':<32} {'RMSE':>7} {'vs base':>8} {'night':>7} {'rush':>7} {'neg':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline_rmse = None
    for name, recipe in build_recipes().items():
        recipe.fit(train_features, train_target)
        predicted = recipe.predict(test_features)

        overall = rmse(test_target, predicted)
        if baseline_rmse is None:
            baseline_rmse = overall

        print(
            f"  {name:<32} {overall:>7.1f} {1 - overall / baseline_rmse:>7.0%}"
            f" {rmse(test_target[night], predicted[night]):>7.1f}"
            f" {rmse(test_target[rush], predicted[rush]):>7.1f}"
            f" {int((predicted < 0).sum()):>5}"
        )


if __name__ == "__main__":
    main()
