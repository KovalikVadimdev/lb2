"""Evaluation of a trained model on one data set.

All models (Hopfield network, Elman network, multilayer perceptron) are
evaluated by the same function, so their errors and classification times
are directly comparable.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from recurrent_lab import config
from recurrent_lab.utils import classification_error, measure_mean_time


class Classifier(Protocol):
    """Any model that returns raw outputs and class labels in one call."""

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating a model on one data set.

    Attributes:
        outputs: Raw model outputs, shape (n, k). For the Elman network and
            the perceptron k is the number of classes; for the Hopfield
            network it is the recalled pattern (one value per pixel).
        labels: Predicted classes 0..9, shape (n,).
        error: Share of misclassified samples (0.0 to 1.0).
        classification_time: Mean time in seconds to classify the whole set.
    """

    outputs: np.ndarray
    labels: np.ndarray
    error: float
    classification_time: float


def evaluate(
    model: Classifier,
    X: np.ndarray,
    y: np.ndarray,
    repeats: int = config.TIMING_REPEATS,
) -> Evaluation:
    """Evaluate a trained model on inputs `X` and labels `y` (0..9).

    The classification time is the mean of `repeats` runs of `model.predict`
    on the whole set, after one warm-up run (see `measure_mean_time`).

    Raises:
        ValueError: If `repeats` is less than 1 or the labels are invalid.
    """
    outputs, labels = model.predict(X)
    error = classification_error(y, labels)
    classification_time = measure_mean_time(lambda: model.predict(X), repeats)

    return Evaluation(
        outputs=outputs,
        labels=labels,
        error=error,
        classification_time=classification_time,
    )
