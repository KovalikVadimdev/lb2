"""Hopfield network used as an associative memory for digit patterns.

The network has one neuron per pixel. Every neuron is connected to all the
others through a symmetric weight matrix. The matrix is computed once from
the stored prototype patterns (there are no epochs). To recognize an image,
the network is started from that image and updates its neurons until the
state stops changing; the state settles near one of the stored prototypes.
"""

import numpy as np

from recurrent_lab import config
from recurrent_lab.data import make_prototypes
from recurrent_lab.utils import Timer, TrainingResult


class HopfieldNetwork:
    """Discrete Hopfield network with bipolar (+1 / -1) neurons.

    Args:
        rule: Learning rule, "hebb" or "pseudoinverse".
        max_iterations: Maximum number of recall passes over all neurons.

    Raises:
        ValueError: If the rule is unknown.
    """

    def __init__(
        self,
        rule: str = "hebb",
        max_iterations: int = config.HOPFIELD_MAX_ITERATIONS,
    ) -> None:
        if rule not in config.HOPFIELD_RULES:
            raise ValueError(
                f"Unknown rule {rule!r}, expected one of {config.HOPFIELD_RULES}"
            )
        self.rule = rule
        self.max_iterations = max_iterations
        self._weights: np.ndarray | None = None
        self._prototypes: np.ndarray | None = None

    @classmethod
    def from_weights(
        cls,
        rule: str,
        weights: np.ndarray,
        prototypes: np.ndarray,
        max_iterations: int = config.HOPFIELD_MAX_ITERATIONS,
    ) -> "HopfieldNetwork":
        """Build a trained network from saved weights and prototypes.

        Args:
            rule: Learning rule the weights were computed with.
            weights: Weight matrix, shape (n_pixels, n_pixels).
            prototypes: Stored patterns, shape (n_classes, n_pixels).
            max_iterations: Maximum number of recall passes.

        Raises:
            ValueError: If the rule is unknown or the shapes do not fit.
        """
        weights = np.asarray(weights, dtype=float)
        prototypes = np.asarray(prototypes, dtype=float)
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError(f"weights must be a square matrix, got {weights.shape}")
        if prototypes.ndim != 2 or prototypes.shape[1] != weights.shape[0]:
            raise ValueError(
                f"prototypes must have shape (n_classes, {weights.shape[0]}), "
                f"got {prototypes.shape}"
            )
        network = cls(rule, max_iterations)
        network._weights = weights.copy()
        network._prototypes = prototypes.copy()
        return network

    @property
    def weights(self) -> np.ndarray:
        """The symmetric weight matrix, shape (n_pixels, n_pixels)."""
        if self._weights is None:
            raise RuntimeError("The network is not trained: call fit first")
        return self._weights

    @property
    def prototypes(self) -> np.ndarray:
        """The stored prototype patterns, shape (n_classes, n_pixels)."""
        if self._prototypes is None:
            raise RuntimeError("The network is not trained: call fit first")
        return self._prototypes

    def fit(self, patterns: np.ndarray, labels: np.ndarray) -> TrainingResult:
        """Build the prototypes from labeled images and compute the weights.

        The prototypes (one bipolar pattern per digit, see `make_prototypes`)
        are built from the bipolar training images `patterns` with integer
        `labels`. Hebb rule: W = (1 / N) * sum of outer products p p^T, with
        a zero diagonal. Pseudoinverse rule: W = P P^+ where the prototypes
        are the columns of P; this is a projection onto the span of the
        prototypes, and the diagonal is kept so that every prototype stays an
        exact fixed point.

        Returns:
            A training result with zero epochs. Its time covers building the
            prototypes and computing the weights.
        """
        with Timer() as timer:
            prototypes = make_prototypes(np.asarray(patterns, dtype=float), labels)
            n_pixels = prototypes.shape[1]
            if self.rule == "hebb":
                weights = prototypes.T @ prototypes / n_pixels
                np.fill_diagonal(weights, 0.0)
            else:
                columns = prototypes.T
                weights = columns @ np.linalg.pinv(columns)
                weights = (weights + weights.T) / 2  # Remove rounding asymmetry.

        self._weights = weights
        self._prototypes = prototypes
        return TrainingResult(
            epochs_run=0,
            stopped_early=False,
            training_time=timer.elapsed,
            loss_history=[],
        )

    def recall(self, patterns: np.ndarray) -> np.ndarray:
        """Recall stored patterns from bipolar start states.

        Neurons are updated one after another (asynchronously) in a fixed
        order, for all samples at once. A pass over all neurons is one
        iteration. Recall stops when a whole pass changes nothing or after
        `max_iterations`. A neuron whose input sum is exactly zero keeps its
        state. The input array is not changed.

        Returns:
            The final states, shape (n, n_pixels).
        """
        weights = self.weights
        states = np.array(patterns, dtype=float)
        n_pixels = states.shape[1]

        for _ in range(self.max_iterations):
            changed = False
            for neuron in range(n_pixels):
                total = states @ weights[:, neuron]
                new_values = np.where(
                    total > 0, 1.0, np.where(total < 0, -1.0, states[:, neuron])
                )
                if np.any(new_values != states[:, neuron]):
                    changed = True
                    states[:, neuron] = new_values
            if not changed:
                break

        return states

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recall the patterns and give each one the label of its prototype.

        The label is the digit whose prototype has the largest dot product
        with the recalled state (the smallest Hamming distance).

        Returns:
            The recalled states (n, n_pixels) and the labels (n,).
        """
        recalled = self.recall(X)
        labels = np.argmax(recalled @ self.prototypes.T, axis=1)
        return recalled, labels
