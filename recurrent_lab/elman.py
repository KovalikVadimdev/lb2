"""Elman network: a simple recurrent network trained by backpropagation
through time.

The network reads an image row by row. At every step the hidden layer gets
the current row and its own state from the previous step (the context).
After the last row the output layer decides which digit it is. The model
computes outputs and gradients, while the weight corrections are applied by
an optimizer from `optimizers.py`.
"""

import numpy as np

from recurrent_lab import config
from recurrent_lab.data import one_hot_bipolar
from recurrent_lab.optimizers import get_optimizer
from recurrent_lab.utils import Timer, TrainingResult, outputs_to_labels


def clip_gradients(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    """Scale the gradients down if their common norm exceeds `max_norm`.

    The norm is computed over all arrays together. If it is not larger than
    `max_norm`, the gradients are returned unchanged; otherwise all of them
    are multiplied by the same factor, so their direction is kept.
    """
    norm = float(np.sqrt(sum(np.sum(grad**2) for grad in grads)))
    if norm <= max_norm:
        return grads
    scale = max_norm / norm
    return [grad * scale for grad in grads]


class ElmanNetwork:
    """Elman network: recurrent hidden layer and an output layer.

    The parameters are stored in the list
    `params = [Wxh, Whh, bh, Why, by]`:
        Wxh:    (n_inputs, hidden_units)        input -> hidden
        Whh:    (hidden_units, hidden_units)    context (previous hidden state)
                                                -> hidden
        bh:     (hidden_units,)
        Why:    (hidden_units, n_outputs)       last hidden state -> output
        by:     (n_outputs,)

    The hidden state is h_t = tanh(x_t Wxh + h_(t-1) Whh + bh) with h_0 = 0,
    and the output is y = tanh(h_T Why + by). The targets are bipolar
    one-hot vectors (+1 for the true digit, -1 for the others).

    Args:
        optimizer: Name of the optimizer (see `optimizers.py`).
        learning_rate: Step size of the optimizer.
        n_inputs: Number of values per time step (pixels in one row).
        hidden_units: Number of neurons in the hidden layer.
        n_outputs: Number of output neurons (classes).
        grad_clip: Largest allowed norm of the gradients in `fit`.
        seed: Seed for the initial weights and for shuffling.

    Raises:
        ValueError: If a size is not positive, the optimizer is unknown or
            the learning rate is not positive.
    """

    def __init__(
        self,
        optimizer: str,
        learning_rate: float = config.DEFAULT_LEARNING_RATE,
        n_inputs: int = config.IMAGE_SIZE,
        hidden_units: int = config.ELMAN_HIDDEN_UNITS,
        n_outputs: int = config.N_CLASSES,
        grad_clip: float = config.ELMAN_GRAD_CLIP,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        if min(n_inputs, hidden_units, n_outputs) < 1:
            raise ValueError("n_inputs, hidden_units and n_outputs must be at least 1")

        self.optimizer_name = optimizer
        self.optimizer = get_optimizer(optimizer, learning_rate)
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        # The spread 1 / sqrt(number of layer inputs) keeps the weighted sums
        # moderate, so the neurons do not start in the saturated zone.
        self.params: list[np.ndarray] = [
            self._rng.normal(0.0, n_inputs**-0.5, (n_inputs, hidden_units)),
            self._rng.normal(0.0, hidden_units**-0.5, (hidden_units, hidden_units)),
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

        The list matches `params`: [dWxh, dWhh, dbh, dWhy, dby]. It is public
        to allow the numerical gradient check.
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
        parameters seen during training are restored at the end. The
        gradients are clipped to `grad_clip` before every update.

        Args:
            X: Sequences scaled to 0..1, shape (n, steps, n_inputs).
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
                    grads = clip_gradients(grads, self.grad_clip)
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
        """Forward pass through all time steps.

        Returns (states, outputs). `states` has the shape
        (n, steps + 1, hidden_units): states[:, 0] is the zero start state
        and states[:, t + 1] is the hidden state after step t. `outputs` has
        the shape (n, n_outputs).
        """
        Wxh, Whh, bh, Why, by = self.params
        n_samples, n_steps, _ = X.shape
        states = np.zeros((n_samples, n_steps + 1, Wxh.shape[1]))
        for step in range(n_steps):
            states[:, step + 1] = np.tanh(X[:, step] @ Wxh + states[:, step] @ Whh + bh)
        outputs = np.tanh(states[:, -1] @ Why + by)
        return states, outputs

    def _gradients(self, X: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
        """Backpropagation through time: gradients of `loss` for all parameters.

        The error of the output is passed to the last hidden state and then
        back through the steps: at every step the hidden state gets its
        blame from the step after it (through Whh) and passes a part of it
        to the input weights, the context weights and the previous state.
        Dividing by n turns the sums over the batch into means.
        """
        _, Whh, _, Why, _ = self.params
        states, outputs = self._forward(X)
        n_steps = X.shape[1]

        d_out = (outputs - targets) / len(X) * (1.0 - outputs**2)
        d_Why = states[:, -1].T @ d_out
        d_by = d_out.sum(axis=0)

        d_Wxh = np.zeros_like(self.params[0])
        d_Whh = np.zeros_like(Whh)
        d_bh = np.zeros_like(self.params[2])

        d_state = d_out @ Why.T
        for step in reversed(range(n_steps)):
            d_sum = d_state * (1.0 - states[:, step + 1] ** 2)
            d_Wxh += X[:, step].T @ d_sum
            d_Whh += states[:, step].T @ d_sum
            d_bh += d_sum.sum(axis=0)
            d_state = d_sum @ Whh.T

        return [d_Wxh, d_Whh, d_bh, d_Why, d_by]

    def _epoch_loss(self, X: np.ndarray, targets: np.ndarray) -> float:
        """Return 0.5 * mean over samples of the summed squared errors."""
        outputs = self._forward(X)[1]
        return float(0.5 * np.mean(np.sum((outputs - targets) ** 2, axis=1)))

    def _check_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Check that X and y have the shapes the network expects."""
        n_inputs = self.params[0].shape[0]
        n_outputs = self.params[3].shape[1]
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3-dimensional (samples, steps, inputs), "
                f"got {X.ndim} dimensions"
            )
        if X.shape[2] != n_inputs:
            raise ValueError(
                f"X has {X.shape[2]} values per step, the network expects {n_inputs}"
            )
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows, but y has {len(y)} labels")
        if np.min(y) < 0 or np.max(y) >= n_outputs:
            raise ValueError(f"Labels must be integers from 0 to {n_outputs - 1}")
