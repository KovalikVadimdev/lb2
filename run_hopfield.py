"""Group A: Hopfield network with the Hebb and pseudoinverse rules.

Two parts:
1.  Main runs. The network is trained with every rule from
    `config.HOPFIELD_RULES` on the real (noise-free) images and the results
    are saved to `results/runs/`.
2.  Noise study. The trained network recalls images with a growing share of
    flipped pixels (`config.NOISE_LEVELS`); the errors are averaged over
    several noise seeds and saved to `results/hopfield_noise_study.csv`.

The network has no epochs: its weights are computed in one step from the
prototype of every digit. The noise is added only at recognition time.

Run from the project root: python run_hopfield.py
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from recurrent_lab import config
from recurrent_lab.data import DigitsSplit, add_noise, load_digits_split, to_bipolar
from recurrent_lab.hopfield import HopfieldNetwork
from recurrent_lab.runner import RunSummary, run_experiment
from recurrent_lab.utils import classification_error

NOISE_STUDY_FILENAME = "hopfield_noise_study.csv"


def prepare_split() -> DigitsSplit:
    """Load the digits and convert the images to bipolar patterns."""
    split = load_digits_split()
    return replace(
        split,
        X_train=to_bipolar(split.X_train),
        X_test=to_bipolar(split.X_test),
    )


def run_group(
    results_dir: Path = config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> list[RunSummary]:
    """Run the Hopfield network with every learning rule and save the runs."""
    split = prepare_split()

    summaries = []
    for rule in config.HOPFIELD_RULES:
        summary = run_experiment(
            lambda: HopfieldNetwork(rule),
            split,
            repeats=repeats,
            results_dir=results_dir,
        )
        summaries.append(summary)
        print_summary(summary)
    return summaries


def run_noise_study(
    results_dir: Path = config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> pd.DataFrame:
    """Measure the error of the trained network for every noise level.

    For every rule the network is trained once on the clean images. For
    every noise level the noise is added to the training and the test
    images `repeats` times with different seeds (the seed is the same for
    both rules, so they see the same noisy images). The mean and the
    standard deviation of the error over the repeats are saved to a CSV file.

    Returns:
        The table that was saved.
    """
    split = prepare_split()

    rows = []
    for rule in config.HOPFIELD_RULES:
        network = HopfieldNetwork(rule)
        network.fit(split.X_train, split.y_train)
        for level in config.NOISE_LEVELS:
            train_errors, test_errors = [], []
            for repeat in range(repeats):
                rng = np.random.default_rng(config.RANDOM_SEED + repeat)
                noisy_train = add_noise(split.X_train, level, rng)
                noisy_test = add_noise(split.X_test, level, rng)
                _, train_labels = network.predict(noisy_train)
                _, test_labels = network.predict(noisy_test)
                train_errors.append(classification_error(split.y_train, train_labels))
                test_errors.append(classification_error(split.y_test, test_labels))
            rows.append(
                {
                    "rule": rule,
                    "noise_level": level,
                    "train_error_mean": np.mean(train_errors),
                    "train_error_std": np.std(train_errors),
                    "test_error_mean": np.mean(test_errors),
                    "test_error_std": np.std(test_errors),
                }
            )

    table = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / NOISE_STUDY_FILENAME, index=False)
    return table


def print_summary(s: RunSummary) -> None:
    """Print one line for a run: training time and errors."""
    print(
        f"{s.run_id}: time={s.training.training_time * 1000:.2f} ms, "
        f"train error={s.train_eval.error:.2%}, "
        f"test error={s.test_eval.error:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    run_group()
    print()
    noise_table = run_noise_study()
    print(noise_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
