"""Small helpers shared by all experiments: timing, labels and error."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from recurrent_lab import config


class Timer:
    """Context manager that measures elapsed wall-clock time.

    After the `with` block ends, `elapsed` holds the duration in seconds,
    measured with `time.perf_counter`.
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, error_type, error_val, traceback) -> None:
        self.elapsed = time.perf_counter() - self._start


def measure_mean_time(
    func: Callable[[], object],
    repeats: int = config.TIMING_REPEATS,
) -> float:
    """Return the mean execution time of `func` in seconds.

    `func` is called once before measuring (warm-up), then `repeats` times
    with timing. It must be safe to call repeatedly, so use it for fast
    operations such as prediction, not for training. Training is timed
    with a single run of `Timer`.

    Raises:
        ValueError: If `repeats` is less than 1.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    func()

    total = 0.0
    for _ in range(repeats):
        with Timer() as timer:
            func()
        total += timer.elapsed

    return total / repeats


def outputs_to_labels(outputs: np.ndarray) -> np.ndarray:
    """Convert model outputs of shape (n, N_CLASSES) to class labels.

    The predicted class is the output with the largest value in each row.
    """
    return np.argmax(outputs, axis=1)


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the share of misclassified samples (0.0 to 1.0).

    Both arrays must contain integer labels from 0 to N_CLASSES - 1.

    Raises:
        ValueError: If the arrays are empty, differ in length, or contain
            values outside the label range.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if y_true.size == 0:
        raise ValueError("Cannot compute error for empty arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.size}, y_pred has {y_pred.size}"
        )

    valid_labels = range(config.N_CLASSES)
    if not (
        np.isin(y_true, valid_labels).all() and np.isin(y_pred, valid_labels).all()
    ):
        raise ValueError(f"Labels must be integers from 0 to {config.N_CLASSES - 1}")

    return float(np.mean(y_true != y_pred))


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one training run.

    A model without epochs (the Hopfield network) reports `epochs_run=0`,
    `stopped_early=False` and an empty loss history.

    Attributes:
        epochs_run: Number of epochs actually performed.
        stopped_early: True if training stopped before `max_epochs`.
        training_time: Training time in seconds.
        loss_history: Training loss after each epoch.
    """

    epochs_run: int
    stopped_early: bool
    training_time: float
    loss_history: list[float]
