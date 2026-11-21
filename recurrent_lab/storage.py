"""Saving a trained model and its results to disk.

Every run gets its own folder `results/runs/<run_id>/` with:
    config.json structure, activations, training method, settings
    metrics.json times, errors, number of epochs
    initial_params.npz parameters before training (if the model has them)
    trained_params.npz parameters after training
    predictions_train.csv inputs, true class, outputs and predicted class
    predictions_test.csv the same for the test set

The models know nothing about files. This module describes them in one
common way, so the same code can save any model type.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recurrent_lab import config
from recurrent_lab.data import DigitsSplit
from recurrent_lab.elman import ElmanNetwork
from recurrent_lab.evaluation import Evaluation
from recurrent_lab.hopfield import HopfieldNetwork
from recurrent_lab.mlp import MultilayerPerceptron
from recurrent_lab.utils import TrainingResult

RUNS_DIRNAME = "runs"
DATASET_NAME = "sklearn digits, 8x8 images, 10 classes"


@dataclass(frozen=True)
class ModelDescription:
    """Common description of any model, used for saving and for the table.

    Attributes:
        kind: "hopfield", "elman" or "mlp".
        n_layers: Number of layers (the input is not counted).
        neurons: Neurons per layer as text, e.g. "64".
        activations: Activation functions per layer as text, e.g. "sign".
        training_method: Learning rule, optimizer or training scheme.
        input_description: How the model input is encoded.
        settings: Model settings (learning rate, seed, ...).
        parameters: Trained parameters by name.
        initial_parameters: Parameters before training, or None if the
            model has no such notion (the Hopfield network).
    """

    kind: str
    n_layers: int
    neurons: str
    activations: str
    training_method: str
    input_description: str = ""
    settings: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    initial_parameters: dict | None = None


def describe_model(model) -> ModelDescription:
    """Describe a model of any supported type in the common form.

    Raises:
        TypeError: If the model type is not supported.
    """
    if isinstance(model, HopfieldNetwork):
        return ModelDescription(
            kind="hopfield",
            n_layers=1,
            neurons=str(model.weights.shape[0]),
            activations="sign",
            training_method=model.rule,
            input_description="bipolar image, 64 pixels (+1 ink, -1 background)",
            settings={
                "max_iterations": model.max_iterations,
                "n_prototypes": len(model.prototypes),
            },
            parameters={"weights": model.weights, "prototypes": model.prototypes},
            initial_parameters=None,
        )
    if isinstance(model, ElmanNetwork):
        names = ("Wxh", "Whh", "bh", "Why", "by")
        n_inputs, hidden_units = model.params[0].shape
        n_outputs = model.params[3].shape[1]
        return ModelDescription(
            kind="elman",
            n_layers=2,
            neurons=f"{hidden_units}-{n_outputs}",
            activations="tanh, tanh",
            training_method=f"bptt-{model.optimizer_name}",
            input_description=(
                f"image rows as {config.IMAGE_SIZE} steps of {n_inputs} pixels, "
                "scaled to 0..1"
            ),
            settings={
                "learning_rate": model.learning_rate,
                "seed": model.seed,
                "n_inputs": n_inputs,
                "hidden_units": hidden_units,
                "sequence_length": config.IMAGE_SIZE,
                "grad_clip": model.grad_clip,
            },
            parameters=dict(zip(names, model.params)),
            initial_parameters=dict(zip(names, model.initial_params)),
        )
    if isinstance(model, MultilayerPerceptron):
        names = ("W1", "b1", "W2", "b2")
        n_inputs, hidden_units = model.params[0].shape
        n_outputs = model.params[2].shape[1]
        return ModelDescription(
            kind="mlp",
            n_layers=2,
            neurons=f"{hidden_units}-{n_outputs}",
            activations="tanh, tanh",
            training_method=model.optimizer_name,
            input_description=f"flat image, {n_inputs} pixels scaled to 0..1",
            settings={
                "learning_rate": model.learning_rate,
                "seed": model.seed,
                "n_inputs": n_inputs,
                "hidden_units": hidden_units,
            },
            parameters=dict(zip(names, model.params)),
            initial_parameters=dict(zip(names, model.initial_params)),
        )

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def build_run_id(description: ModelDescription, tag: str = "") -> str:
    """Build a readable unique name of a run.

    Example: "hopfield_64_sign_hebb". The learning rate and the seed are
    added when the model has them, e.g. "elman_16-10_tanh-tanh_adam_lr0.01_seed42".
    The optional `tag` is appended at the end.
    """
    parts = [
        description.kind,
        description.neurons,
        description.activations.replace(", ", "-"),
        description.training_method,
    ]
    learning_rate = description.settings.get("learning_rate")
    if learning_rate is not None:
        parts.append(f"lr{learning_rate:g}")
    seed = description.settings.get("seed")
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(tag)
    return "_".join(parts)


def save_run(
    run_id: str,
    model,
    training: TrainingResult,
    train_eval: Evaluation,
    test_eval: Evaluation,
    split: DigitsSplit,
    training_settings: dict | None = None,
    split_seed: int = config.RANDOM_SEED,
    results_dir: Path = config.RESULTS_DIR,
) -> Path:
    """Save everything about one run and return its folder.

    An existing folder with the same `run_id` is overwritten.

    Args:
        run_id: Name of the run, see `build_run_id`.
        model: A trained model of a supported type.
        training: Result of `model.fit`.
        train_eval: Evaluation on the training set.
        test_eval: Evaluation on the test set.
        split: The model inputs the model was trained and tested on.
        training_settings: Arguments of `fit` (batch size, patience, ...).
        split_seed: Seed of the train/test split.
        results_dir: Root folder for the results.
    """
    description = describe_model(model)
    run_dir = results_dir / RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "kind": description.kind,
        "n_layers": description.n_layers,
        "neurons": description.neurons,
        "activations": description.activations,
        "training_method": description.training_method,
        "settings": description.settings,
        "training_settings": training_settings or {},
        "data": {
            "dataset": DATASET_NAME,
            "input": description.input_description,
            "train_samples": len(split.X_train),
            "test_samples": len(split.X_test),
            "split_seed": split_seed,
            "test_size": config.TEST_SIZE,
        },
    }
    metrics = {
        "training_time_s": training.training_time,
        "epochs_run": training.epochs_run,
        "stopped_early": training.stopped_early,
        "final_training_loss": (
            training.loss_history[-1] if training.loss_history else None
        ),
        "train": {
            "error": train_eval.error,
            "classification_time_s": train_eval.classification_time,
        },
        "test": {
            "error": test_eval.error,
            "classification_time_s": test_eval.classification_time,
        },
    }
    _write_json(run_dir / "config.json", run_config)
    _write_json(run_dir / "metrics.json", metrics)

    if description.initial_parameters is not None:
        np.savez(run_dir / "initial_params.npz", **description.initial_parameters)
    np.savez(run_dir / "trained_params.npz", **description.parameters)

    _write_predictions(
        run_dir / "predictions_train.csv", split.X_train, split.y_train, train_eval
    )
    _write_predictions(
        run_dir / "predictions_test.csv", split.X_test, split.y_test, test_eval
    )
    return run_dir


def _write_predictions(
    path: Path, X: np.ndarray, y: np.ndarray, evaluation: Evaluation
) -> None:
    """Write the model inputs, true classes, outputs and predicted classes.

    Inputs of any shape are flattened to one row of pixels per sample.
    """
    inputs = X.reshape(len(X), -1)
    outputs = evaluation.outputs.reshape(len(X), -1)
    table = pd.concat(
        [
            pd.DataFrame(
                inputs, columns=[f"pixel_{i}" for i in range(inputs.shape[1])]
            ),
            pd.DataFrame({"true_class": y, "predicted_class": evaluation.labels}),
            pd.DataFrame(
                outputs, columns=[f"out_{j}" for j in range(outputs.shape[1])]
            ),
        ],
        axis=1,
    )
    table.to_csv(path, index=False)


def _write_json(path: Path, data: dict) -> None:
    """Write a dictionary as readable JSON (numpy values are converted)."""
    path.write_text(
        json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _to_jsonable(value):
    """Convert numpy values to plain Python and non-finite numbers to None."""
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
