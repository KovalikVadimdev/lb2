"""Build the summary tables of all saved runs.

Reads `config.json` and `metrics.json` of every folder in `results/runs/`
and writes two files to `results/`:
    summary.csv     all runs, English column names (for further processing)
    summary.xlsx    Ukrainian column names (for the report), two sheets:
                    the main table (the Hopfield network and the networks
                    with the default hidden layer size) and the hidden
                    layer size study

Units: training time in seconds (3 significant digits), classification time
in milliseconds, errors in percent.

Run from the project root: python build_summary.py
"""

import json
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font

from recurrent_lab import config
from recurrent_lab.storage import RUNS_DIRNAME

ARCHITECTURE_NAMES = {
    "hopfield": ("Hopfield network", "Мережа Хопфілда"),
    "elman": ("Elman network", "Мережа Елмана"),
    "mlp": ("Multilayer perceptron", "Багатошаровий перцептрон (базова мережа)"),
}
METHOD_NAMES_UK = {
    "hebb": "Правило Хеба",
    "pseudoinverse": "Псевдообернена матриця",
    "sgd": "SGD",
    "adam": "Adam",
    "bptt-sgd": "BPTT + SGD",
    "bptt-adam": "BPTT + Adam",
}
ACTIVATION_NAMES_UK = {"sign": "Знакова (sign)"}
DEFAULT_HIDDEN_UNITS = {
    "elman": config.ELMAN_HIDDEN_UNITS,
    "mlp": config.MLP_HIDDEN_UNITS,
}
LEARNING_RATES = {
    "elman": config.ELMAN_LEARNING_RATES,
    "mlp": config.MLP_LEARNING_RATES,
}

MAIN_SHEET = "Основна таблиця"
STUDY_SHEET = "Розміри прихованого шару"

# (column in the CSV, column in the Excel file). The first ten columns are
# the table required by the lab methodology, the rest is extra information.
TABLE_COLUMNS = [
    ("architecture", "Назва архітектури"),
    ("n_layers", "Кількість шарів"),
    ("neurons", "Кількість нейронів у шарах"),
    ("activations", "Функції активації"),
    ("training_method", "Метод навчання"),
    ("training_time_s", "Час навчання, с"),
    ("train_classification_time_ms", "Час класифікації (train), мс"),
    ("train_error_pct", "Помилка (train), %"),
    ("test_classification_time_ms", "Час класифікації (test), мс"),
    ("test_error_pct", "Помилка (test), %"),
    ("learning_rate", "Швидкість навчання"),
    ("epochs", "Кількість епох"),
    ("stopped_early", "Зупинка за критерієм"),
]
CSV_ONLY_COLUMNS = ["run_id"]


def load_runs(results_dir: Path = config.RESULTS_DIR) -> pd.DataFrame:
    """Collect all saved runs into one table, one row per run.

    Raises:
        FileNotFoundError: If there are no saved runs.
    """
    runs_dir = results_dir / RUNS_DIRNAME
    folders = [p for p in runs_dir.iterdir() if p.is_dir()] if runs_dir.is_dir() else []
    if not folders:
        raise FileNotFoundError(f"No runs found in {runs_dir}")

    rows = []
    for folder in folders:
        run_config = _read_json(folder / "config.json")
        metrics = _read_json(folder / "metrics.json")
        kind = run_config["kind"]
        # The Hopfield network has no epochs, so these two fields do not apply.
        has_epochs = kind != "hopfield"
        rows.append(
            {
                "kind": kind,
                "architecture": ARCHITECTURE_NAMES[kind][0],
                "n_layers": run_config["n_layers"],
                "neurons": run_config["neurons"],
                "activations": run_config["activations"],
                "training_method": run_config["training_method"],
                "training_time_s": _three_significant(metrics["training_time_s"]),
                "train_classification_time_ms": round(
                    metrics["train"]["classification_time_s"] * 1000, 3
                ),
                "train_error_pct": round(metrics["train"]["error"] * 100, 2),
                "test_classification_time_ms": round(
                    metrics["test"]["classification_time_s"] * 1000, 3
                ),
                "test_error_pct": round(metrics["test"]["error"] * 100, 2),
                "learning_rate": run_config["settings"].get("learning_rate"),
                "epochs": metrics["epochs_run"] if has_epochs else None,
                "stopped_early": metrics["stopped_early"] if has_epochs else None,
                "hidden_units": (
                    int(run_config["neurons"].split("-")[0]) if has_epochs else None
                ),
                "run_id": run_config["run_id"],
            }
        )

    table = pd.DataFrame(rows)
    # Whole numbers with gaps (the Hopfield network has no epochs).
    table["epochs"] = table["epochs"].astype("Int64")
    order = [_sort_key(row) for row in rows]
    return table.iloc[sorted(range(len(rows)), key=order.__getitem__)].reset_index(
        drop=True
    )


def split_main_and_study(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the runs into the main table and the hidden layer size study.

    The main table holds the Hopfield network and the networks with the
    default hidden layer size. The study holds the networks with other sizes,
    plus the default-size networks trained in the same way (same kind,
    method and learning rate), so the sizes can be compared in one place.
    """
    default_size = table["kind"].map(DEFAULT_HIDDEN_UNITS)
    is_main = table["kind"].eq("hopfield") | table["hidden_units"].eq(default_size)
    other = table[~is_main]

    study_setups = set(
        zip(other["kind"], other["training_method"], other["learning_rate"])
    )
    is_reference = [
        is_main_row and (kind, method, rate) in study_setups
        for is_main_row, kind, method, rate in zip(
            is_main, table["kind"], table["training_method"], table["learning_rate"]
        )
    ]
    study = table[~is_main | pd.Series(is_reference, index=table.index)]
    kind_rank = study["kind"].map(list(ARCHITECTURE_NAMES).index)
    study = study.assign(_rank=kind_rank).sort_values(
        ["_rank", "hidden_units"], kind="stable"
    )
    return table[is_main], study.drop(columns="_rank")


def write_csv(table: pd.DataFrame, path: Path) -> None:
    """Write all runs with English column names."""
    columns = [name for name, _ in TABLE_COLUMNS] + CSV_ONLY_COLUMNS
    table[columns].to_csv(path, index=False)


def write_excel(table: pd.DataFrame, path: Path) -> None:
    """Write the report table with Ukrainian names, one sheet per group."""
    main, study = split_main_and_study(table)
    sheets = {MAIN_SHEET: main}
    if not study.empty:
        sheets[STUDY_SHEET] = study

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, part in sheets.items():
            view = _to_report_view(part)
            view.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_sheet(writer.sheets[sheet_name], view)


def _to_report_view(part: pd.DataFrame) -> pd.DataFrame:
    """Return the rows with Ukrainian names and values for the report."""
    view = part.copy()
    view["architecture"] = view["kind"].map(lambda k: ARCHITECTURE_NAMES[k][1])
    view["activations"] = view["activations"].map(
        lambda a: ACTIVATION_NAMES_UK.get(a, a)
    )
    view["training_method"] = view["training_method"].map(
        lambda m: METHOD_NAMES_UK.get(m, m)
    )
    view["stopped_early"] = view["stopped_early"].map({True: "Так", False: "Ні"})
    columns = dict(TABLE_COLUMNS)
    return view[list(columns)].rename(columns=columns)


def _format_sheet(sheet, view: pd.DataFrame) -> None:
    """Bold wrapped header, frozen first row and readable column widths."""
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.freeze_panes = "A2"
    for index, name in enumerate(view.columns, start=1):
        longest = max([len(str(name)) // 2] + [len(str(v)) for v in view[name]])
        letter = sheet.cell(row=1, column=index).column_letter
        sheet.column_dimensions[letter].width = min(max(longest + 2, 10), 40)


def _sort_key(row: dict) -> tuple:
    """Order of the rows: by model type, then in the order of the config."""
    kind_rank = list(ARCHITECTURE_NAMES).index(row["kind"])
    if row["kind"] == "hopfield":
        return (kind_rank, 0, config.HOPFIELD_RULES.index(row["training_method"]), 0.0)

    optimizer = row["training_method"].removeprefix("bptt-")
    optimizers = list(LEARNING_RATES[row["kind"]])
    # Default size first, then the other sizes from small to large.
    other_size = int(row["hidden_units"] != DEFAULT_HIDDEN_UNITS[row["kind"]])
    return (
        kind_rank,
        other_size,
        row["hidden_units"] if other_size else optimizers.index(optimizer),
        row["learning_rate"],
    )


def _three_significant(value: float) -> float:
    """Round to three significant digits (0.000243 stays readable)."""
    return float(f"{value:.3g}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(results_dir: Path = config.RESULTS_DIR) -> None:
    """Build `summary.csv` and `summary.xlsx` from the saved runs."""
    table = load_runs(results_dir)
    csv_path = results_dir / "summary.csv"
    xlsx_path = results_dir / "summary.xlsx"
    write_csv(table, csv_path)
    write_excel(table, xlsx_path)

    main_part, study_part = split_main_and_study(table)
    print(
        f"Runs: {len(table)} (main sheet: {len(main_part)}, size study: {len(study_part)})"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")


if __name__ == "__main__":
    main()
