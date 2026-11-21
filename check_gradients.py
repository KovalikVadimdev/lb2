"""Check the backpropagation-through-time gradients of the Elman network.

For every parameter value the script changes it by a tiny step in both
directions and measures how the loss changes. This measured slope must
match the analytical gradient of the model.

Run from the project root: python check_gradients.py
The check passes if every difference is below the threshold.
"""

import numpy as np

from recurrent_lab import config
from recurrent_lab.elman import ElmanNetwork
from recurrent_lab.mlp import MultilayerPerceptron

STEP = 1e-6
THRESHOLD = 1e-6
N_SAMPLES = 5
SEED = 0
ELMAN_PARAM_NAMES = ("Wxh", "Whh", "bh", "Why", "by")
MLP_PARAM_NAMES = ("W1", "b1", "W2", "b2")


def gradient_differences(model, X, y) -> list[float]:
    """Return the largest |numerical - analytical| difference per parameter."""
    gradients = model.compute_gradients(X, y)
    differences = []

    for param, gradient in zip(model.params, gradients):
        if param.shape != gradient.shape:
            raise ValueError(
                f"Gradient shape {gradient.shape} differs from "
                f"parameter shape {param.shape}"
            )
        worst = 0.0
        for index in np.ndindex(param.shape):
            original = param[index]

            param[index] = original + STEP
            loss_up = model.loss(X, y)
            param[index] = original - STEP
            loss_down = model.loss(X, y)
            param[index] = original

            numerical = (loss_up - loss_down) / (2 * STEP)
            worst = max(worst, abs(numerical - gradient[index]))
        differences.append(worst)

    return differences


def report(title: str, model, param_names, X, y) -> bool:
    """Print the difference of one model and return True if all are small."""
    print(title)
    passed_all = True
    for name, difference in zip(param_names, gradient_differences(model, X, y)):
        passed = difference < THRESHOLD
        passed_all = passed_all and passed
        label = "OK" if passed else "CHECK"
        print(f"{name:4s} max difference: {difference:.2e} [{label}]")
    return passed_all


def main() -> None:
    """Run the check on random inputs with randomly initialized networks."""
    rng = np.random.default_rng(SEED)
    side = config.IMAGE_SIZE
    sequences = rng.uniform(0.0, 1.0, size=(N_SAMPLES, side, side))
    y = rng.integers(0, config.N_CLASSES, size=N_SAMPLES)

    elman = ElmanNetwork(config.ELMAN_OPTIMIZERS[0], seed=1)
    mlp = MultilayerPerceptron(config.MLP_OPTIMIZERS[0], seed=1)

    results = [
        report("Elman network", elman, ELMAN_PARAM_NAMES, sequences, y),
        report(
            "Multilayer perceptron",
            mlp,
            MLP_PARAM_NAMES,
            sequences.reshape(N_SAMPLES, -1),
            y,
        ),
    ]

    print("All checks passed" if all(results) else "Some checks FAILED")


if __name__ == "__main__":
    main()
