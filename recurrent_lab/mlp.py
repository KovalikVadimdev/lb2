"""Multilayer perceptron with one hidden layer, trained by backpropagation.

This is the feedforward network used for comparison with the recurrent
networks. It sees the whole image at once (all pixels as one input vector),
without reading it step by step and without any context. The model computes
outputs and gradients, while the weight corrections are applied by an
optimizer from `optimizers.py`.
"""

import numpy as np

from recurrent_lab import config
from recurrent_lab.data import one_hot_bipolar
from recurrent_lab.optimizers import get_optimizer
from recurrent_lab.utils import Timer, TrainingResult, outputs_to_labels


class MultilayerPerceptron:
    """Two-layer perceptron: a tanh hidden layer and a tanh output layer.

    The parameters are stored in the list `params = [W1, b1, W2, b2]`:
        W1: (n_inputs, hidden_units),   b1: (hidden_units,)
        W2: (hidden_units, n_outputs),  b2: (n_outputs,)

    The targets are bipolar one-hot vectors (+1 for the true digit, -1 for
    the others), the same as for the Elman network.

    Args:
        optimizer: Name of the optimizer (see `optimizers.py`).
        learning_rate: Step size of the optimizer.
        n_inputs: Number of input values (pixels of the whole image).
        hidden_units: Number of neurons in the hidden layer.
        n_outputs: Number of output neurons (classes).
        seed: Seed for the initial weights and for shuffling.

    Raises:
        ValueError: If a size is not positive, the optimizer is unknown or
            the learning rate is not positive.
    """

    def __init__(
        self,
        optimizer: str,
        learning_rate: float = config.DEFAULT_LEARNING_RATE,
        n_inputs: int = config.IMAGE_SIZE**2,
        hidden_units: int = config.MLP_HIDDEN_UNITS,
        n_outputs: int = config.N_CLASSES,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        if min(n_inputs, hidden_units, n_outputs) < 1:
            raise ValueError("n_inputs, hidden_units and n_outputs must be at least 1")

        self.optimizer_name = optimizer
        self.optimizer = get_optimizer(optimizer, learning_rate)
        self.learning_rate = learning_rate
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        # The spread 1 / sqrt(number of layer inputs) keeps the weighted sums
        # moderate, so the neurons do not start in the saturated zone.
        self.params: list[np.ndarray] = [
            self._rng.normal(0.0, n_inputs**-0.5, (n_inputs, hidden_units)),
            np.zeros(hidden_units),
            self._rng.normal(0.0, hidden_units**-0.5, (hidden_units, n_outputs)),
            np.zeros(n_outputs),
        ]

        # Kept for the report: the state before any training.
        self.initial_params = [param.copy() for param in self.params]

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw outputs (n, n_outputs) and the class labels (n,)."""
        outputs = self._forward(X)[1]
        return outputs, outputs_to_labels(outputs)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return 0.5 * mean over samples of the summed squared errors."""
        return self._epoch_loss(X, one_hot_bipolar(y))

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Return the gradients of `loss` for labels `y` (0..9).

        The list matches `params`: [dW1, db1, dW2, db2]. It is public to
        allow the numerical gradient check.
        """
        return self._gradients(X, one_hot_bipolar(y))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int | None = config.BATCH_SIZE,
        max_epochs: int = config.MAX_EPOCHS,
        patience: int = config.PATIENCE,
        min_delta: float = config.MIN_DELTA,
    ) -> TrainingResult:
        """Train the network with early stopping.

        Training stops when the loss is zero, or has not improved by at least
        `min_delta` for `patience` epochs in a row, or when the loss is not a
        finite number (divergence), or after `max_epochs`. The best
        parameters seen during training are restored at the end.

        Args:
            X: Images scaled to 0..1, shape (n, n_inputs).
            y: Labels 0..9, shape (n,).
            batch_size: Samples per weight update. None means the whole set.
            max_epochs: Upper limit of epochs.
            patience: Epochs without improvement before stopping.
            min_delta: Smallest loss decrease counted as an improvement.

        Returns:
            A `TrainingResult`. `stopped_early` is False if the epoch limit was
            reached or the training diverged.

        Raises:
            ValueError: If the shapes of `X` and `y` do not fit the network
                or `batch_size` is not positive.
        """
        self._check_training_data(X, y)
        targets = one_hot_bipolar(y)

        n_samples = len(X)
        if batch_size is None:
            batch_size = n_samples
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        loss_history: list[float] = []
        best_loss = np.inf
        best_params = [param.copy() for param in self.params]
        wait = 0
        stopped_early = False

        with Timer() as timer:
            for _ in range(max_epochs):
                order = self._rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    idx = order[start : start + batch_size]
                    grads = self._gradients(X[idx], targets[idx])
                    self.optimizer.update(self.params, grads)

                loss = self._epoch_loss(X, targets)
                loss_history.append(loss)

                if not np.isfinite(loss):
                    break

                if loss < best_loss - min_delta:
                    best_loss = loss
                    best_params = [param.copy() for param in self.params]
                    wait = 0
                else:
                    wait += 1

                if loss == 0.0 or wait >= patience:
                    stopped_early = True
                    break

        # Restore in place, so the arrays in `self.params` stay the same objects.
        for param, best in zip(self.params, best_params):
            param[...] = best

        return TrainingResult(
            epochs_run=len(loss_history),
            stopped_early=stopped_early,
            training_time=timer.elapsed,
            loss_history=loss_history,
        )

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (hidden outputs, outputs).

        The hidden outputs have the shape (n, hidden_units) and are needed
        by backpropagation; the outputs have the shape (n, n_outputs).
        """
        W1, b1, W2, b2 = self.params
        hidden = np.tanh(X @ W1 + b1)
        outputs = np.tanh(hidden @ W2 + b2)
        return hidden, outputs

    def _gradients(self, X: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
        """Backward pass: gradients of `loss` for all parameters.

        Dividing by n turns the sums over the batch into means. Every
        gradient has the shape of its parameter.
        """
        _, _, W2, _ = self.params
        hidden, outputs = self._forward(X)

        d_out = (outputs - targets) / len(X) * (1.0 - outputs**2)
        d_W2 = hidden.T @ d_out
        d_b2 = d_out.sum(axis=0)

        # The blame of the output error is passed to the hidden neurons
        # through the weights of the second layer.
        d_hidden = (d_out @ W2.T) * (1.0 - hidden**2)
        d_W1 = X.T @ d_hidden
        d_b1 = d_hidden.sum(axis=0)

        return [d_W1, d_b1, d_W2, d_b2]

    def _epoch_loss(self, X: np.ndarray, targets: np.ndarray) -> float:
        """Return 0.5 * mean over samples of the summed squared errors."""
        outputs = self._forward(X)[1]
        return float(0.5 * np.mean(np.sum((outputs - targets) ** 2, axis=1)))

    def _check_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Check that X and y have the shapes the network expects."""
        n_inputs = self.params[0].shape[0]
        n_outputs = self.params[2].shape[1]
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        if X.shape[1] != n_inputs:
            raise ValueError(
                f"X has {X.shape[1]} values per sample, the network expects {n_inputs}"
            )
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows, but y has {len(y)} labels")
        if np.min(y) < 0 or np.max(y) >= n_outputs:
            raise ValueError(f"Labels must be integers from 0 to {n_outputs - 1}")
