# -*- coding: utf-8 -*-
"""
make_mlx_bar_from_5_csv.py

Purpose:
1. Read five epoch_metrics.csv files.
2. Aggregate mean/error data for a 1x4 MLX-style bar chart.
3. Export CSV files that MATLAB/MLX can read directly.
4. Render the same 1x4 bar chart as PNG and PDF.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline_suite.config.paths import (
    BAR_CSV_CONFIGS,
    BAR_LAST20_OUTPUT_DIR,
    BAR_MATLAB_ARRAYS_FILE,
    BAR_PYTHON_FIG_PDF,
    BAR_PYTHON_FIG_PNG,
    BAR_SUMMARY_LONG_CSV,
    BAR_SUMMARY_WIDE_CSV,
    BAR_TRUNCATED_DIR,
    BAR_TRUNCATED_MANIFEST,
    BAR_TRUNCATED_SUFFIX,
)


# ============================================================
# Analysis behavior. Input and output paths are defined in config/paths.py.
# ============================================================

CSV_CONFIGS = BAR_CSV_CONFIGS
OUTPUT_DIR: Optional[str] = None
DEFAULT_OUTPUT_DIR = BAR_LAST20_OUTPUT_DIR

# Aggregation modes: last_n, all, final, or best.
AGGREGATE_MODE = "last_n"
LAST_N_EPOCHS = 20

# Error modes: standard deviation, standard error, or no error bars.
ERROR_MODE = "std"

# Convert completion rates in [0, 1] to percentages automatically.
AUTO_PERCENT_COMPLETION = True

# Candidate violation columns, ordered by preference.
VIOLATION_COLUMN_CANDIDATES = [
    "num_violations",
    "violations",
    "violation",
    "task_violations",
    "num_failed",
]

# Candidate source columns for the four reported metrics.
METRIC_COLUMNS = {
    "completion": ["completion_rate", "task_completion_rate", "complete_rate"],
    "latency": ["avg_latency_s", "task_latency", "latency", "avg_delay_s"],
    "energy": ["energy_proxy", "task_energy", "energy", "avg_energy"],
    "violation": VIOLATION_COLUMN_CANDIDATES,
}

SUMMARY_WIDE_CSV = BAR_SUMMARY_WIDE_CSV
SUMMARY_LONG_CSV = BAR_SUMMARY_LONG_CSV
MATLAB_ARRAYS_M = BAR_MATLAB_ARRAYS_FILE
PYTHON_FIG_PNG = BAR_PYTHON_FIG_PNG
PYTHON_FIG_PDF = BAR_PYTHON_FIG_PDF

# Optionally export the last N rows for each method with aligned epoch numbers.
EXPORT_TRUNCATED_LAST_N_CSV = True
TRUNCATED_LAST_N_EPOCHS = 20
TRUNCATED_CSV_DIR_NAME = BAR_TRUNCATED_DIR
TRUNCATED_CSV_SUFFIX = BAR_TRUNCATED_SUFFIX



# ============================================================
# Analysis helpers.
# ============================================================

def resolve_output_dir() -> Path:
    if OUTPUT_DIR is not None and str(OUTPUT_DIR).strip():
        out_dir = Path(OUTPUT_DIR).expanduser().resolve()
    else:
        try:
            out_dir = DEFAULT_OUTPUT_DIR
        except NameError:
            out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], metric_name: str) -> str:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in col_map:
            return col_map[key]
    raise KeyError(
        f"No column found for {metric_name}. Candidates={list(candidates)}, actual={list(df.columns)}"
    )


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def select_epoch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "epoch" in df.columns:
        df = df.copy()
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values("epoch", kind="mergesort")

    mode = str(AGGREGATE_MODE).lower().strip()

    if mode == "all":
        return df

    if mode == "final":
        return df.tail(1)

    if mode == "best":
        if "is_best" in df.columns:
            b = df["is_best"]
            if b.dtype == bool:
                best_df = df[b]
            else:
                best_df = df[b.astype(str).str.lower().isin(["true", "1", "yes", "y"])]
            if not best_df.empty:
                return best_df
        return df.tail(1)

    if mode == "last_n":
        n = int(LAST_N_EPOCHS) if LAST_N_EPOCHS is not None else len(df)
        n = max(1, n)
        return df.tail(n)

    raise ValueError(f"Unknown AGGREGATE_MODE={AGGREGATE_MODE!r}; expected all/last_n/final/best")


def mean_and_error(values: pd.Series) -> Tuple[float, float]:
    arr = np.asarray(values.dropna(), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    mode = str(ERROR_MODE).lower().strip()
    if mode == "none" or arr.size == 1:
        err = 0.0
    elif mode == "sem":
        err = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    elif mode == "std":
        err = float(np.std(arr, ddof=1))
    else:
        raise ValueError(f"Unknown ERROR_MODE={ERROR_MODE!r}; expected std/sem/none")
    return mean, err


def maybe_to_percent_completion(s: pd.Series) -> pd.Series:
    if not AUTO_PERCENT_COMPLETION or s.empty:
        return s
    q95 = float(s.quantile(0.95)) if len(s) else 0.0
    if q95 <= 1.5:
        return s * 100.0
    return s


def summarize_one_csv(method: str, path: str) -> Dict[str, object]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    used_df = select_epoch_rows(df)

    row: Dict[str, object] = {
        "method": method,
        "source_csv": str(csv_path),
        "num_epochs_total": int(len(df)),
        "num_epochs_used": int(len(used_df)),
        "epoch_start": "",
        "epoch_end": "",
    }

    if "epoch" in used_df.columns and not used_df.empty:
        ep = pd.to_numeric(used_df["epoch"], errors="coerce").dropna()
        if not ep.empty:
            row["epoch_start"] = int(ep.min())
            row["epoch_end"] = int(ep.max())

    for metric, candidates in METRIC_COLUMNS.items():
        col = find_first_existing_column(used_df, candidates, metric)
        s = numeric_series(used_df, col)
        if metric == "completion":
            s = maybe_to_percent_completion(s)
        mean, err = mean_and_error(s)
        row[f"{metric}_column"] = col
        row[f"{metric}_mean"] = mean
        row[f"{metric}_err"] = err

    return row


def build_summary() -> pd.DataFrame:
    if len(CSV_CONFIGS) != 5:
        raise ValueError(f"Configured {len(CSV_CONFIGS)} CSV files; exactly five are required.")
    rows = []
    seen_methods = set()
    for item in CSV_CONFIGS:
        method = str(item.get("method", "")).strip()
        path = str(item.get("path", "")).strip()
        if not method:
            raise ValueError(f"Empty method name: {item}")
        if method in seen_methods:
            raise ValueError(f"Duplicate method name: {method}")
        seen_methods.add(method)
        if not path or path.startswith("/path/to/"):
            raise ValueError(f"Provide the real CSV path for {method}.")
        rows.append(summarize_one_csv(method, path))
    return pd.DataFrame(rows)


def to_long_table(wide: pd.DataFrame) -> pd.DataFrame:
    records = []
    metric_labels = {
        "completion": "Complete Rate (%) (higher is better)",
        "latency": "Latency / mission (lower is better)",
        "energy": "Energy / mission (lower is better)",
        "violation": "# Violation (lower is better)",
    }
    for _, r in wide.iterrows():
        for metric in ["completion", "latency", "energy", "violation"]:
            records.append({
                "metric": metric,
                "metric_label": metric_labels[metric],
                "method": r["method"],
                "mean": r[f"{metric}_mean"],
                "err": r[f"{metric}_err"],
                "source_csv": r["source_csv"],
                "num_epochs_used": r["num_epochs_used"],
                "epoch_start": r["epoch_start"],
                "epoch_end": r["epoch_end"],
            })
    return pd.DataFrame(records)


def matlab_string_cell(items: Sequence[str]) -> str:
    escaped = [str(x).replace("'", "''") for x in items]
    return "{" + ", ".join(f"'{x}'" for x in escaped) + "}"


def matlab_array(values: Sequence[float]) -> str:
    return "[" + ", ".join("nan" if pd.isna(v) else f"{float(v):.6g}" for v in values) + "]"


def write_matlab_arrays_file(wide: pd.DataFrame, out_path: Path) -> None:
    methods = list(wide["method"].astype(str))
    text = f"""%% Auto-generated by make_mlx_bar_from_5_csv.py
% You can run this file before your MLX plotting section, or replace the arrays in the MLX.

methods = {matlab_string_cell(methods)};
nMethods = numel(methods);

completeMean = {matlab_array(wide['completion_mean'])};
completeErr  = {matlab_array(wide['completion_err'])};

latencyMean  = {matlab_array(wide['latency_mean'])};
latencyErr   = {matlab_array(wide['latency_err'])};

energyMean   = {matlab_array(wide['energy_mean'])};
energyErr    = {matlab_array(wide['energy_err'])};

violationMean = {matlab_array(wide['violation_mean'])};
violationErr  = {matlab_array(wide['violation_err'])};
"""
    out_path.write_text(text, encoding="utf-8")



def safe_filename(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "method"


def select_last_n_rows_for_export(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "epoch" in out.columns:
        out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
        out = out.sort_values("epoch", kind="mergesort")
    n = max(1, int(n))
    return out.tail(n).copy()


def export_truncated_last_n_csvs(out_dir: Path) -> List[Path]:
    """Export one truncated CSV per method.

    Each exported CSV keeps all original columns, takes only the last
    TRUNCATED_LAST_N_EPOCHS rows after sorting by epoch, and rewrites the
    epoch column to 1,2,...,N so MATLAB/MLX can read aligned curves directly.
    """
    if not bool(EXPORT_TRUNCATED_LAST_N_CSV):
        return []

    trunc_dir = out_dir / TRUNCATED_CSV_DIR_NAME
    trunc_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: List[Path] = []
    manifest_rows: List[Dict[str, object]] = []

    for idx, item in enumerate(CSV_CONFIGS, start=1):
        method = str(item.get("method", "")).strip()
        csv_path = Path(str(item.get("path", "")).strip()).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV does not exist: {csv_path}")

        df = pd.read_csv(csv_path)
        used_df = select_last_n_rows_for_export(df, TRUNCATED_LAST_N_EPOCHS)
        original_epoch_start = ""
        original_epoch_end = ""
        if "epoch" in used_df.columns and not used_df.empty:
            ep = pd.to_numeric(used_df["epoch"], errors="coerce").dropna()
            if not ep.empty:
                original_epoch_start = int(ep.min())
                original_epoch_end = int(ep.max())

        if "epoch" in used_df.columns:
            used_df["epoch"] = np.arange(1, len(used_df) + 1, dtype=int)
        else:
            used_df.insert(0, "epoch", np.arange(1, len(used_df) + 1, dtype=int))

        file_name = f"{idx:02d}_{safe_filename(method)}_{TRUNCATED_CSV_SUFFIX}"
        out_path = trunc_dir / file_name
        used_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        exported_paths.append(out_path)

        manifest_rows.append({
            "method": method,
            "source_csv": str(csv_path),
            "truncated_csv": str(out_path),
            "num_rows_source": int(len(df)),
            "num_rows_exported": int(len(used_df)),
            "original_epoch_start": original_epoch_start,
            "original_epoch_end": original_epoch_end,
            "renamed_epoch_start": 1 if len(used_df) > 0 else "",
            "renamed_epoch_end": int(len(used_df)) if len(used_df) > 0 else "",
        })

        if len(used_df) < int(TRUNCATED_LAST_N_EPOCHS):
            print(
                f"[warning] {method} has only {len(used_df)} rows, fewer than the requested {TRUNCATED_LAST_N_EPOCHS}.",
                flush=True,
            )

    manifest_path = trunc_dir / BAR_TRUNCATED_MANIFEST
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    exported_paths.append(manifest_path)
    return exported_paths

def write_outputs(wide: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path, Path]:
    wide_path = out_dir / SUMMARY_WIDE_CSV
    long_path = out_dir / SUMMARY_LONG_CSV
    m_path = out_dir / MATLAB_ARRAYS_M

    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    to_long_table(wide).to_csv(long_path, index=False, encoding="utf-8-sig")
    write_matlab_arrays_file(wide, m_path)
    return wide_path, long_path, m_path


def plot_four_panel(wide: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    methods = list(wide["method"].astype(str))
    x = np.arange(len(methods), dtype=float)

    panels = [
        ("completion", "Complete Rate (%) (higher is better)"),
        ("latency", "Latency / mission (lower is better)"),
        ("energy", "Energy / mission (lower is better)"),
        ("violation", "# Violation (lower is better)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.4), constrained_layout=True)

    for ax, (metric, ylabel) in zip(axes, panels):
        means = wide[f"{metric}_mean"].astype(float).to_numpy()
        errs = wide[f"{metric}_err"].astype(float).to_numpy()
        if str(ERROR_MODE).lower().strip() == "none":
            errs = None

        ax.bar(x, means, yerr=errs, capsize=3, width=0.65, edgecolor="black", linewidth=0.8)
        ax.plot(x, means, marker="o", linewidth=1.0, markersize=3.5)

        ymax_values = means if errs is None else means + np.nan_to_num(errs, nan=0.0)
        ymax = float(np.nanmax(ymax_values)) if np.isfinite(ymax_values).any() else 1.0
        ymin = 0.0
        ypad = 0.16 * max(abs(ymax - ymin), 1e-9)
        ax.set_ylim(ymin, ymax + ypad)

        for i, v in enumerate(means):
            if np.isfinite(v):
                ax.text(i, v + 0.035 * max(ymax, 1.0), f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    png_path = out_dir / PYTHON_FIG_PNG
    pdf_path = out_dir / PYTHON_FIG_PDF
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def print_mlx_read_example(wide_csv_path: Path) -> None:
    print("\nRead the prepared CSV in MLX as follows:")
    print("------------------------------------------------------------")
    print(f"T = readtable('{wide_csv_path.as_posix()}', 'TextType', 'string');")
    print("methods = cellstr(T.method);  % Alternatively: methods = T.method';")
    print("completeMean = T.completion_mean';")
    print("completeErr  = T.completion_err';")
    print("latencyMean  = T.latency_mean';")
    print("latencyErr   = T.latency_err';")
    print("energyMean   = T.energy_mean';")
    print("energyErr    = T.energy_err';")
    print("violationMean = T.violation_mean';")
    print("violationErr  = T.violation_err';")
    print("------------------------------------------------------------\n")


def main() -> None:
    out_dir = resolve_output_dir()
    wide = build_summary()

    wide_path, long_path, m_path = write_outputs(wide, out_dir)
    truncated_paths = export_truncated_last_n_csvs(out_dir)
    png_path, pdf_path = plot_four_panel(wide, out_dir)

    print("Completed. Output files:")
    print(f"1) MLX wide CSV : {wide_path}")
    print(f"2) MLX long CSV : {long_path}")
    print(f"3) MATLAB arrays: {m_path}")
    print(f"4) Python PNG   : {png_path}")
    print(f"5) Python PDF   : {pdf_path}")
    if truncated_paths:
        print("6) Truncated last-20 epoch CSV files:")
        for p in truncated_paths:
            print(f"   - {p}")
    print_mlx_read_example(wide_path)
    print("Summary preview:")
    print(wide[[
        "method", "num_epochs_used", "epoch_start", "epoch_end",
        "completion_mean", "completion_err",
        "latency_mean", "latency_err",
        "energy_mean", "energy_err",
        "violation_mean", "violation_err",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
