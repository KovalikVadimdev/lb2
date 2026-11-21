"""Data preparation for the digit recognition task.

The images come from the 8x8 handwritten digits set shipped with
scikit-learn. This module loads them, splits them into training and test
parts, and converts them to the forms the networks need: bipolar patterns
(+1 / -1) for the Hopfield network and sequences of rows for the Elman
network. It also adds noise and builds one prototype pattern per digit.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from recurrent_lab import config

# The largest pixel value in the digits set (pixels range from 0 to 16).
MAX_PIXEL_VALUE = 16.0


@dataclass
class DigitsSplit:
    """Training and test parts of the digits set.

    Images are flat rows of IMAGE_SIZE ** 2 raw pixel values (0 to 16).
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_digits_split() -> DigitsSplit:
    """Load the digits and split them into stratified train and test sets.

    The same seed always gives the same split.
    """
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=digits.target,
    )
    return DigitsSplit(X_train, X_test, y_train, y_test)


def to_bipolar(X: np.ndarray) -> np.ndarray:
    """Turn raw pixels into a bipolar pattern: +1 for ink, -1 for background."""
    return np.where(X > config.BINARIZE_THRESHOLD, 1.0, -1.0)


def to_unit_range(X: np.ndarray) -> np.ndarray:
    """Scale raw pixel values from 0..16 to 0..1."""
    return X / MAX_PIXEL_VALUE


def to_sequences(X: np.ndarray) -> np.ndarray:
    """Reshape flat images into sequences of image rows.

    Input shape (n, 64) becomes (n, 8, 8): the second axis is the time step
    (the row index), the third axis holds the pixels of that row.
    """
    return X.reshape(len(X), config.IMAGE_SIZE, config.IMAGE_SIZE)


def one_hot_bipolar(y: np.ndarray) -> np.ndarray:
    """Build bipolar one-hot targets: +1 for the true class, -1 elsewhere."""
    targets = -np.ones((len(y), config.N_CLASSES))
    targets[np.arange(len(y)), y] = 1.0
    return targets


def add_noise(
    patterns: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    """Flip a share of the pixels in every bipolar pattern.

    Exactly round(level * n_pixels) randomly chosen pixels are inverted in
    each pattern. A new array is returned; the input is not changed.
    """
    noisy = patterns.copy()
    n_pixels = patterns.shape[1]
    n_flips = int(round(level * n_pixels))
    if n_flips == 0:
        return noisy
    for row in noisy:
        positions = rng.choice(n_pixels, size=n_flips, replace=False)
        row[positions] *= -1
    return noisy


def make_prototypes(patterns: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build one bipolar prototype per class.

    A prototype is the sign of the mean pattern of the class; a mean of
    exactly zero becomes +1. The result has shape (N_CLASSES, n_pixels), and
    row k is the prototype of digit k.
    """
    prototypes = np.empty((config.N_CLASSES, patterns.shape[1]))
    for digit in range(config.N_CLASSES):
        mean_pattern = patterns[labels == digit].mean(axis=0)
        prototypes[digit] = np.where(mean_pattern >= 0, 1.0, -1.0)
    return prototypes
