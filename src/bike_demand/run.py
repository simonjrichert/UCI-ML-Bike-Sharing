"""One command: train the same model under both splits and compare."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_hours, features_and_target, timestamps
from .evaluate import report, rmse, score_split
from .models import SHORT_LABELS, build_recipes
from .splits import Split, random_split, time_split
from .train import BASELINE, MODEL

PLOT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "time_split_forecast.png"
ZOOM_HOURS = 24 * 14


def plot_time_split(split: Split, path: Path = PLOT_PATH) -> Path:
    """Draw the held-out hours no variant saw, actual against every prediction."""
    train_features, train_target = features_and_target(split.train)
    test_features, actual = features_and_target(split.test)
    when = timestamps(split.test)

    predictions = {}
    for name, recipe in build_recipes().items():
        recipe.fit(train_features, train_target)
        predictions[name] = recipe.predict(test_features)

    figure, (full, zoom) = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    window = slice(0, ZOOM_HOURS)

    for axis, rows, width in ((full, slice(None), 0.7), (zoom, window, 1.1)):
        axis.plot(
            when[rows], actual[rows], linewidth=width + 0.4, color="black", label="actual"
        )
        for name, predicted in predictions.items():
            axis.plot(
                when[rows],
                predicted[rows],
                linewidth=width,
                alpha=0.85,
                label=f"{SHORT_LABELS[name]} (RMSE {rmse(actual, predicted):.0f})",
            )
        axis.set_ylabel("rentals per hour")

    full.set_title(
        "Held-out hours no variant trained on (Nov-Dec 2012), with all-period RMSE"
    )
    full.legend(loc="upper right", ncol=3, fontsize=8)
    zoom.set_title("First two weeks, hour by hour")
    zoom.set_xlabel("date")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    hours = load_hours()
    print(f"{len(hours):,} hours loaded, {hours['dteday'].min().date()} to {hours['dteday'].max().date()}\n")

    honest = time_split(hours)
    random_result = score_split(random_split(hours))
    time_result = score_split(honest)

    print(report(random_result), "\n")
    print(report(time_result), "\n")

    optimism = time_result["test_rmse"][MODEL] - random_result["test_rmse"][MODEL]
    print(
        f"Same model, same features, same metric. Shuffling the hours makes the error "
        f"look {optimism:.1f} rentals/hour better than forecasting the real future."
    )
    print(f"Baseline for reference: {time_result['test_rmse'][BASELINE]:.1f} RMSE on the future hours.")
    print("\nFitting all five variants for the plot ...")
    print(f"Plot written to {plot_time_split(honest)}")


if __name__ == "__main__":
    main()
