"""Draw all graphs of the report from the saved results.

Reads the saved runs (through `build_summary.load_runs`),
`results/hopfield_noise_study.csv` (made by `run_hopfield.py`) and the
weights of the saved Hopfield networks, and writes the graphs to
`results/figures/`:
    hopfield_recall.png             stored patterns, noisy inputs and recall
    hopfield_noise.png              Hopfield error against the noise level
    optimizers.png                  Elman network and perceptron, optimizers
    size_study.png                  error and time against the hidden size
    architectures_comparison.png    all runs together

The graphs need no training, so they can be redrawn in a second. All texts
are in the `TEXT` dictionary: change `LANGUAGE` to switch between Ukrainian
("uk") and English ("en").

Run from the project root: python plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from build_summary import DEFAULT_HIDDEN_UNITS, load_runs, split_main_and_study
from recurrent_lab import config
from recurrent_lab.data import add_noise
from recurrent_lab.hopfield import HopfieldNetwork
from run_hopfield import NOISE_STUDY_FILENAME

LANGUAGE = "uk"
EXAMPLE_NOISE_LEVEL = 0.2  # Noise of the recall example figure.

TEXT = {
    "uk": {
        "noise_level": "Частка інвертованих пікселів, %",
        "error": "Помилка, %",
        "test_error": "Помилка на тесті, %",
        "training_time": "Час навчання, с",
        "classification_time": "Час класифікації (test), мс",
        "hidden_units": "Кількість прихованих нейронів",
        "train": "навчальна вибірка",
        "test": "тестова вибірка",
        "stored": "Еталон",
        "noisy": f"Вхід (шум {EXAMPLE_NOISE_LEVEL:.0%})",
        "recall_title": "Відновлення зашумлених еталонів мережею Хопфілда",
        "noise_title": "Мережа Хопфілда: стійкість до шуму",
        "optimizers_error": "Помилка на тесті",
        "optimizers_time": "Час навчання",
        "size_error": "Помилка залежно від розміру прихованого шару",
        "size_time": "Час навчання залежно від розміру прихованого шару",
        "vs_training_time": "Точність і час навчання",
        "vs_classification_time": "Точність і час класифікації",
        "rules": {"hebb": "Хеб", "pseudoinverse": "Псевдообернена матриця"},
        "kinds": {
            "hopfield": "Мережа Хопфілда",
            "elman": "Мережа Елмана",
            "mlp": "Багатошаровий перцептрон",
        },
        "optimizer_names": {"sgd": "SGD", "adam": "Adam"},
    },
    "en": {
        "noise_level": "Share of flipped pixels, %",
        "error": "Error, %",
        "test_error": "Test error, %",
        "training_time": "Training time, s",
        "classification_time": "Classification time (test), ms",
        "hidden_units": "Number of hidden neurons",
        "train": "training set",
        "test": "test set",
        "stored": "Stored",
        "noisy": f"Input ({EXAMPLE_NOISE_LEVEL:.0%} noise)",
        "recall_title": "Recall of noisy patterns by the Hopfield network",
        "noise_title": "Hopfield network: noise robustness",
        "optimizers_error": "Test error",
        "optimizers_time": "Training time",
        "size_error": "Error against the hidden layer size",
        "size_time": "Training time against the hidden layer size",
        "vs_training_time": "Accuracy and training time",
        "vs_classification_time": "Accuracy and classification time",
        "rules": {"hebb": "Hebb", "pseudoinverse": "Pseudoinverse"},
        "kinds": {
            "hopfield": "Hopfield network",
            "elman": "Elman network",
            "mlp": "Multilayer perceptron",
        },
        "optimizer_names": {"sgd": "SGD", "adam": "Adam"},
    },
}
KIND_STYLES = {
    "hopfield": ("tab:green", "o"),
    "elman": ("tab:blue", "s"),
    "mlp": ("tab:orange", "^"),
}
RULE_COLORS = {"hebb": "tab:red", "pseudoinverse": "tab:green"}
ERROR_TICKS = [2, 3, 5, 10, 20, 50, 90]  # Readable ticks of the log error axis.
TIME_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10]  # Ticks of the classification time axis.


def _text(key: str):
    return TEXT[LANGUAGE][key]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _optimizer_of(method: str) -> str:
    """Return the optimizer name of a training method ("bptt-adam" -> "adam")."""
    return method.removeprefix("bptt-")


def _load_hopfield(results_dir: Path, rule: str) -> HopfieldNetwork:
    """Restore a saved Hopfield network from its weights file."""
    path = results_dir / "runs" / f"hopfield_64_sign_{rule}" / "trained_params.npz"
    with np.load(path) as params:
        return HopfieldNetwork.from_weights(
            rule, params["weights"], params["prototypes"]
        )


def plot_hopfield_recall(results_dir: Path, path: Path) -> None:
    """Show the stored patterns, their noisy copies and what each rule recalls."""
    networks = {
        rule: _load_hopfield(results_dir, rule) for rule in config.HOPFIELD_RULES
    }
    prototypes = networks[config.HOPFIELD_RULES[0]].prototypes
    rng = np.random.default_rng(config.RANDOM_SEED)
    noisy = add_noise(prototypes, EXAMPLE_NOISE_LEVEL, rng)

    rows = [(_text("stored"), prototypes), (_text("noisy"), noisy)]
    for rule, network in networks.items():
        rows.append((_text("rules")[rule], network.recall(noisy)))

    side = config.IMAGE_SIZE
    fig, axes = plt.subplots(len(rows), config.N_CLASSES, figsize=(11, 4.8))
    for row_index, (label, patterns) in enumerate(rows):
        for digit in range(config.N_CLASSES):
            axis = axes[row_index, digit]
            axis.imshow(
                patterns[digit].reshape(side, side), cmap="gray_r", vmin=-1, vmax=1
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(str(digit))
        axes[row_index, 0].set_ylabel(label, rotation=0, ha="right", va="center")
    fig.suptitle(_text("recall_title"))
    fig.subplots_adjust(
        left=0.2, right=0.99, top=0.88, bottom=0.02, wspace=0.08, hspace=0.08
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hopfield_noise(results_dir: Path, path: Path) -> None:
    """Draw the error against the noise level, one color per learning rule."""
    study = pd.read_csv(results_dir / NOISE_STUDY_FILENAME)
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for rule in config.HOPFIELD_RULES:
        part = study[study["rule"] == rule].sort_values("noise_level")
        x = part["noise_level"] * 100
        color = RULE_COLORS[rule]
        name = _text("rules")[rule]
        axis.errorbar(
            x,
            part["test_error_mean"] * 100,
            part["test_error_std"] * 100,
            color=color,
            marker="o",
            capsize=3,
            label=f"{name}, {_text('test')}",
        )
        axis.errorbar(
            x,
            part["train_error_mean"] * 100,
            part["train_error_std"] * 100,
            color=color,
            marker="o",
            capsize=3,
            linestyle="--",
            label=f"{name}, {_text('train')}",
        )
    axis.set_xlabel(_text("noise_level"))
    axis.set_ylabel(_text("error"))
    axis.set_title(_text("noise_title"))
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.3)
    axis.legend()
    _save(fig, path)


def _setup_value(
    part: pd.DataFrame, kind: str, optimizer: str, rate: float, column: str
):
    """Return `column` of the one run with this optimizer and learning rate.

    Raises:
        ValueError: If there is no such run (the experiment was not run).
    """
    match = part[
        (part["training_method"].map(_optimizer_of) == optimizer)
        & (part["learning_rate"] == rate)
    ]
    if match.empty:
        raise ValueError(
            f"No saved run of the {kind} network with {optimizer} and learning "
            f"rate {rate:g}: run its experiment script first"
        )
    return match[column].iloc[0]


def plot_optimizers(table: pd.DataFrame, path: Path) -> None:
    """Compare the optimizer settings of the Elman network and the perceptron."""
    default = table["hidden_units"].eq(table["kind"].map(DEFAULT_HIDDEN_UNITS))
    runs = table[default & table["kind"].isin(["elman", "mlp"])]

    setups = [
        (optimizer, rate)
        for optimizer, rates in config.ELMAN_LEARNING_RATES.items()
        for rate in rates
    ]
    labels = [
        f"{_text('optimizer_names')[optimizer]}\n{rate:g}" for optimizer, rate in setups
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    columns = [
        ("test_error_pct", "optimizers_error", "test_error"),
        ("training_time_s", "optimizers_time", "training_time"),
    ]
    width = 0.38
    for axis, (column, title_key, label_key) in zip(axes, columns):
        for offset, kind in ((-width / 2, "elman"), (width / 2, "mlp")):
            part = runs[runs["kind"] == kind]
            values = [
                _setup_value(part, kind, optimizer, rate, column)
                for optimizer, rate in setups
            ]
            axis.bar(
                np.arange(len(setups)) + offset,
                values,
                width,
                color=KIND_STYLES[kind][0],
                label=_text("kinds")[kind],
            )
        axis.set_xticks(np.arange(len(setups)), labels)
        axis.set_ylabel(_text(label_key))
        axis.set_title(_text(title_key))
        axis.set_ylim(0, axis.get_ylim()[1] * 1.2)  # Room for the legend.
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right")
    _save(fig, path)


def plot_size_study(table: pd.DataFrame, path: Path) -> None:
    """Draw the error and the training time against the hidden layer size."""
    _, study = split_main_and_study(table)
    sizes = sorted(study["hidden_units"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for kind in ("elman", "mlp"):
        part = study[study["kind"] == kind].sort_values("hidden_units")
        color, marker = KIND_STYLES[kind]
        name = _text("kinds")[kind]
        axes[0].plot(
            part["hidden_units"],
            part["test_error_pct"],
            color=color,
            marker=marker,
            label=f"{name}, {_text('test')}",
        )
        axes[0].plot(
            part["hidden_units"],
            part["train_error_pct"],
            color=color,
            marker=marker,
            linestyle="--",
            label=f"{name}, {_text('train')}",
        )
        axes[1].plot(
            part["hidden_units"],
            part["training_time_s"],
            color=color,
            marker=marker,
            label=name,
        )

    for axis, title_key, label_key in (
        (axes[0], "size_error", "error"),
        (axes[1], "size_time", "training_time"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.set_xlabel(_text("hidden_units"))
        axis.set_ylabel(_text(label_key))
        axis.set_title(_text(title_key))
        axis.grid(alpha=0.3)
        axis.legend()
    _save(fig, path)


def plot_architectures(table: pd.DataFrame, path: Path) -> None:
    """Draw every run: test error against training and classification time."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    columns = [
        ("training_time_s", "training_time", "vs_training_time"),
        (
            "test_classification_time_ms",
            "classification_time",
            "vs_classification_time",
        ),
    ]
    for axis, (column, label_key, title_key) in zip(axes, columns):
        for kind, (color, marker) in KIND_STYLES.items():
            part = table[table["kind"] == kind]
            axis.scatter(
                part[column],
                part["test_error_pct"],
                color=color,
                marker=marker,
                s=45,
                alpha=0.8,
                label=_text("kinds")[kind],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(_text(label_key))
        axis.set_ylabel(_text("test_error"))
        axis.set_title(_text(title_key))
        axis.grid(alpha=0.3, which="both")
        axis.set_yticks(ERROR_TICKS, [str(tick) for tick in ERROR_TICKS])
        axis.yaxis.set_minor_formatter(NullFormatter())
        if column == "test_classification_time_ms":
            axis.set_xticks(TIME_TICKS, [f"{tick:g}" for tick in TIME_TICKS])
    axes[0].legend()
    _save(fig, path)


def main(results_dir: Path = config.RESULTS_DIR) -> list[Path]:
    """Draw all graphs from the saved results and return their paths.

    Raises:
        FileNotFoundError: If the runs or the noise study are not saved yet.
    """
    noise_path = results_dir / NOISE_STUDY_FILENAME
    if not noise_path.is_file():
        raise FileNotFoundError(f"{noise_path} not found: run run_hopfield.py first")
    table = load_runs(results_dir)
    missing = {"hopfield", "elman", "mlp"} - set(table["kind"])
    if missing:
        raise FileNotFoundError(
            f"No saved runs of: {', '.join(sorted(missing))}. Run run_hopfield.py, "
            "run_elman.py and run_mlp_baseline.py first"
        )
    figures = results_dir / "figures"

    paths = {
        "hopfield_recall.png": lambda p: plot_hopfield_recall(results_dir, p),
        "hopfield_noise.png": lambda p: plot_hopfield_noise(results_dir, p),
        "optimizers.png": lambda p: plot_optimizers(table, p),
        "size_study.png": lambda p: plot_size_study(table, p),
        "architectures_comparison.png": lambda p: plot_architectures(table, p),
    }
    saved = []
    for name, draw in paths.items():
        draw(figures / name)
        saved.append(figures / name)
        print(f"Saved: {figures / name}")
    return saved


if __name__ == "__main__":
    main()
