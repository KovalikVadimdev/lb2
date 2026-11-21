"""Group B: Elman network with different optimizers and hidden layer sizes.

Two parts:
1.  Main runs. The network with `config.ELMAN_HIDDEN_UNITS` hidden neurons is
    trained with every optimizer and learning rate from
    `config.ELMAN_LEARNING_RATES`.
2.  Hidden size study. Networks with the sizes from
    `config.ELMAN_HIDDEN_SIZES_STUDY` are trained with Adam and learning
    rate 0.01.

An image is read row by row, so one sample is a sequence of 8 steps with
8 pixels each. All results are saved to `results/runs/`.

Run from the project root: python run_elman.py
"""

from dataclasses import replace
from pathlib import Path

from recurrent_lab import config
from recurrent_lab.data import (
    DigitsSplit,
    load_digits_split,
    to_sequences,
    to_unit_range,
)
from recurrent_lab.elman import ElmanNetwork
from recurrent_lab.runner import RunSummary, run_experiment

HIDDEN_STUDY_OPTIMIZER = "adam"
HIDDEN_STUDY_LEARNING_RATE = 0.01


def prepare_split() -> DigitsSplit:
    """Load the digits and turn the images into sequences of rows (0..1)."""
    split = load_digits_split()
    return replace(
        split,
        X_train=to_sequences(to_unit_range(split.X_train)),
        X_test=to_sequences(to_unit_range(split.X_test)),
    )


def run_group(
    results_dir: Path = config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> list[RunSummary]:
    """Run every optimizer with every learning rate and return the summaries."""
    split = prepare_split()

    summaries = []
    for optimizer, learning_rates in config.ELMAN_LEARNING_RATES.items():
        for learning_rate in learning_rates:
            summary = run_experiment(
                lambda: ElmanNetwork(optimizer, learning_rate),
                split,
                repeats=repeats,
                results_dir=results_dir,
            )
            summaries.append(summary)
            print_summary(summary)
    return summaries


def run_hidden_study(
    results_dir: Path = config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> list[RunSummary]:
    """Run networks with different numbers of hidden neurons."""
    split = prepare_split()

    summaries = []
    for hidden_units in config.ELMAN_HIDDEN_SIZES_STUDY:
        summary = run_experiment(
            lambda: ElmanNetwork(
                HIDDEN_STUDY_OPTIMIZER,
                HIDDEN_STUDY_LEARNING_RATE,
                hidden_units=hidden_units,
            ),
            split,
            repeats=repeats,
            results_dir=results_dir,
        )
        summaries.append(summary)
        print_summary(summary)
    return summaries


def print_summary(s: RunSummary) -> None:
    """Print one line for a run: epochs, training time and errors."""
    print(
        f"{s.run_id}: epochs={s.training.epochs_run}, "
        f"time={s.training.training_time:.2f} s, "
        f"train error={s.train_eval.error:.2%}, "
        f"test error={s.test_eval.error:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    run_group()
    print()
    run_hidden_study()
