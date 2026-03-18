"""
Analysis and visualization of ES vs GRPO parameter norm statistics.

Refactored from a notebook-style script into reusable functions and a single
`main()` entrypoint, while preserving the original analysis behavior.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/ES_vs_GRPO"
)
ES_JSONL = BASE_DIR / "es_param_norms.jsonl"
GRPO_JSONL = BASE_DIR / "grpo_param_norms.jsonl"

# ES random-walk formula:
#   E ||Δθ̄_T||² = σ² T d / (4 N)
ES_SIGMA_DEFAULT = 0.0015
ES_POP_SIZE_DEFAULT = 30
ES_REF_INDEX_DEFAULT = 499  # row used in original script for reference snapshot


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file line by line as dicts."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {count} in {path}: {e}")
                continue

            yield obj
            count += 1
            if max_records is not None and count >= max_records:
                break


def load_sample_as_dataframe(filename: str, max_records: int = 10_000) -> pd.DataFrame:
    """Load up to `max_records` rows from a JSONL file under `BASE_DIR`."""
    path = BASE_DIR / filename
    records = list(iter_jsonl(path, max_records=max_records))
    return pd.DataFrame.from_records(records)


def load_first_param_norms_record(path: Path | str) -> Dict[str, Any]:
    """Return the first JSON record in a param-norms JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    raise RuntimeError(f"{path} is empty or has no valid JSON lines.")


# ---------------------------------------------------------------------------
# Param-norms → DataFrame helpers
# ---------------------------------------------------------------------------

def _shape_to_str(shape: Iterable[int] | None) -> Optional[str]:
    if shape is None:
        return None
    return " x ".join(str(x) for x in shape)


def param_norms_to_df(param_norms: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """
    Flatten a single `param_norms` mapping:

        { "param_name": {"delta_norm_sq": ..., "shape": [...], "numel": ..., ...}, ... }

    into a tidy DataFrame.
    """
    rows: list[MutableMapping[str, Any]] = []
    for name, stats in param_norms.items():
        row: Dict[str, Any] = {"param_name": name, **stats}
        shape = stats.get("shape")
        if isinstance(shape, (list, tuple)):
            row["shape_str"] = _shape_to_str(shape)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "delta_norm_sq" in df.columns:
        df = df.sort_values("delta_norm_sq", ascending=False, ignore_index=True)
    return df


def param_norms_list_to_df(
    param_norms_list: Iterable[Mapping[str, Mapping[str, Any]]]
) -> pd.DataFrame:
    """
    Flatten a list of `param_norms` dicts across steps into a single DataFrame.

    Each element of `param_norms_list` is a `param_norms` dict as in `param_norms_to_df`.
    """
    rows: list[MutableMapping[str, Any]] = []

    for step_idx, param_dict in enumerate(param_norms_list):
        if not isinstance(param_dict, Mapping):
            continue

        for name, stats in param_dict.items():
            if not isinstance(stats, Mapping):
                continue

            row: Dict[str, Any] = {"step": step_idx, "param_name": name, **stats}
            shape = stats.get("shape")
            if isinstance(shape, (list, tuple)):
                row["shape_str"] = _shape_to_str(shape)
            rows.append(row)

    df = pd.DataFrame(rows)
    if {"delta_norm_sq", "step"} <= set(df.columns):
        df = df.sort_values(["step", "delta_norm_sq"], ascending=[True, False], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Param-name parsing / layer & kind
# ---------------------------------------------------------------------------

def add_layer_and_kind_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 'param_name' like:
      - 'model.layers.21.mlp.gate_up_proj.weight'
      - 'model.layers.0.self_attn.q_proj.weight'
      - 'model.embed_tokens.weight' (no layer index)

    extract:
      - 'layer' (int, -1 if no layer)
      - 'kind' (sub-module grouping, e.g. 'mlp.gate_up_proj.weight').
    """

    def _parse(name: str) -> Tuple[int, str]:
        match = re.search(r"\.layers\.(\d+)\.(.+)", name)
        if match:
            layer = int(match.group(1))
            kind = match.group(2)
        else:
            layer = -1
            if name.startswith("model."):
                kind = name[len("model.") :]
            else:
                kind = name
        return layer, kind

    layers: list[int] = []
    kinds: list[str] = []
    for n in df["param_name"]:
        layer, kind = _parse(str(n))
        layers.append(layer)
        kinds.append(kind)

    out = df.copy()
    out["layer"] = layers
    out["kind"] = kinds
    return out


# ---------------------------------------------------------------------------
# ES slope / random-walk analysis
# ---------------------------------------------------------------------------

def build_param_delta_trajectories(es_df: pd.DataFrame) -> Dict[str, list[Tuple[int, float]]]:
    """Collect per-parameter `(step, delta_norm_sq)` trajectories from ES DataFrame."""
    param_delta_dict: Dict[str, list[Tuple[int, float]]] = defaultdict(list)

    for _, row in es_df.iterrows():
        step = int(row["step"])
        norms_df = param_norms_list_to_df(row["param_norms"])
        for _, entry in norms_df.iterrows():
            param_name = entry["param_name"]
            delta_norm_sq = float(entry["delta_norm_sq"])
            param_delta_dict[param_name].append((step, delta_norm_sq))

    return param_delta_dict


def fit_param_slopes(param_delta_dict: Mapping[str, Iterable[Tuple[int, float]]]) -> pd.DataFrame:
    """Fit no-intercept linear regressions of `delta_norm_sq` vs `step` for each parameter."""
    slope_records: list[Dict[str, Any]] = []

    for param_name, data in param_delta_dict.items():
        data_sorted = sorted(data)
        steps, delta_norms = zip(*data_sorted)
        steps_arr = np.asarray(steps, dtype=np.float64).reshape(-1, 1)
        delta_arr = np.asarray(delta_norms, dtype=np.float64)

        reg = LinearRegression(fit_intercept=False)
        reg.fit(steps_arr, delta_arr)

        slope = float(reg.coef_[0])
        y_pred = reg.predict(steps_arr)
        ss_res = float(((delta_arr - y_pred) ** 2).sum())
        ss_tot = float(((delta_arr - delta_arr.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        slope_records.append({"param_name": param_name, "slope": slope, "r2": r2})

    return pd.DataFrame(slope_records).set_index("param_name")


def compute_ref_param_info_slope(
    es_df: pd.DataFrame,
    param_slopes: pd.DataFrame,
    ref_index: int = ES_REF_INDEX_DEFAULT,
    sigma: float = ES_SIGMA_DEFAULT,
    population_size: int = ES_POP_SIZE_DEFAULT,
) -> pd.DataFrame:
    """
    Combine a reference ES param_norms snapshot with slope / R² info and derived metrics:

        d_randwalk ≈ slope / σ² * 4 N
        d_ratio    = d_randwalk / numel
    """
    ref_param_info = param_norms_list_to_df(es_df.iloc[ref_index]["param_norms"])
    ref_param_info = ref_param_info.drop(columns=["delta_norm_sq"], errors="ignore")

    ref_param_info_slope = ref_param_info.merge(
        param_slopes, left_on="param_name", right_index=True, how="left"
    )

    ref_param_info_slope["d_randwalk"] = (
        ref_param_info_slope["slope"] / sigma**2 * 4.0 * population_size
    )
    ref_param_info_slope["d_ratio"] = ref_param_info_slope["d_randwalk"] / ref_param_info_slope["numel"]
    ref_param_info_slope["1-d_ratio"] = 1.0 - ref_param_info_slope["d_ratio"]

    return ref_param_info_slope


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_param_heatmap(
    df: pd.DataFrame,
    value_col: str = "slope",
    step: int = 0,
    include_nonlayer: bool = False,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Heatmap with:
      - rows: parameter kind
      - columns: layer index
      - values: `value_col` (e.g. 'slope', 'd_randwalk', 'd_ratio').
    """
    if "layer" not in df.columns or "kind" not in df.columns:
        df = add_layer_and_kind_columns(df)

    if "step" in df.columns:
        df_step = df[df["step"] == step].copy()
    else:
        df_step = df.copy()

    if not include_nonlayer:
        df_step = df_step[df_step["layer"] >= 0]

    if df_step.empty:
        if "step" in df.columns:
            raise ValueError(f"No data for step={step} after filtering.")
        raise ValueError("No data after filtering (and no 'step' column present).")

    pivot = df_step.pivot_table(
        index="kind",
        columns="layer",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=1).sort_index(axis=0)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cbar_kws={"label": value_col},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Parameter kind")
    suffix = f" (step={step})" if "step" in df.columns else ""
    ax.set_title(f"{value_col} heatmap by layer and param kind{suffix}")
    plt.tight_layout()
    plt.show()


def plot_d_ratio_vs_r2(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["d_ratio"], df["r2"], alpha=0.7)
    plt.xlabel("d_ratio")
    plt.ylabel("$R^2$")
    plt.title("Scatter of d_ratio vs $R^2$ for parameters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_numel_vs_d_ratio(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["numel"], df["d_ratio"], s=10, alpha=0.5)
    plt.xlabel("numel")
    plt.ylabel("d_ratio")
    plt.title("Scatter of numel vs d_ratio")
    plt.xscale("log")
    plt.tight_layout()
    plt.show()


def plot_kind_colored_scatter(df: pd.DataFrame) -> None:
    df_param_norms = add_layer_and_kind_columns(df)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df_param_norms,
        x="d_ratio",
        y="r2",
        hue="kind",
        alpha=0.7,
        palette="tab10",
        s=49,
    )
    plt.xlabel("d_ratio = d_randwalk / numel")
    plt.ylabel("$R^2$")
    plt.title("Scatter of d_ratio vs $R^2$ by param type")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Param Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def plot_scale_vs_linearity(df: pd.DataFrame) -> None:
    df_param_norms = add_layer_and_kind_columns(df)
    numel_log = np.log10(df_param_norms["numel"].replace(0, np.nan))

    # log(numel) vs R^2
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df_param_norms,
        x=numel_log,
        y="r2",
        hue="kind",
        alpha=0.7,
        palette="tab10",
        s=49,
    )
    plt.xlabel("log₁₀(numel)")
    plt.ylabel("$R^2$")
    plt.title("log₁₀(numel) vs $R^2$ for parameters")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Param Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

    # log(numel) vs d_ratio
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df_param_norms,
        x=numel_log,
        y="d_ratio",
        hue="kind",
        alpha=0.7,
        palette="tab10",
        s=49,
    )
    plt.xlabel("log₁₀(numel)")
    plt.ylabel("d_ratio = d_randwalk / numel")
    plt.title("log₁₀(numel) vs d_ratio for parameters")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Param Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


# ---------------------------------------------------------------------------
# GRPO comparison
# ---------------------------------------------------------------------------

def build_grpo_param_norms_snapshot(grpo_df: pd.DataFrame, row_index: int) -> pd.DataFrame:
    df_param_norms = param_norms_to_df(grpo_df.iloc[row_index]["param_norms"])
    df_param_norms = add_layer_and_kind_columns(df_param_norms)
    df_param_norms["delta_norm_sq_per_el"] = df_param_norms["delta_norm_sq"] / df_param_norms["numel"]
    return df_param_norms


def run_grpo_comparison(grpo_df: pd.DataFrame) -> None:
    # Mirrors the original script: use rows 78 and 20 as examples
    df_param_norms_78 = build_grpo_param_norms_snapshot(grpo_df, 78)
    plot_param_heatmap(
        df_param_norms_78.query("numel > 1e5"),
        value_col="delta_norm_sq",
        cmap="coolwarm",
    )

    df_param_norms_20 = build_grpo_param_norms_snapshot(grpo_df, 20)
    plot_param_heatmap(
        df_param_norms_20.query("numel > 1e5"),
        value_col="delta_norm_sq_per_el",
        cmap="coolwarm",
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def preview_raw_records() -> None:
    """Print a quick preview of the raw ES/GRPO JSONL records."""
    print("First ES record:")
    for rec in iter_jsonl(ES_JSONL, max_records=1):
        print(rec)
    print()

    print("First GRPO record:")
    for rec in iter_jsonl(GRPO_JSONL, max_records=1):
        print(rec)
    print()


def load_es_grpo_samples(
    es_max_records: int = 10_000,
    grpo_max_records: int = 10_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    es_df = load_sample_as_dataframe("es_param_norms.jsonl", max_records=es_max_records)
    grpo_df = load_sample_as_dataframe("grpo_param_norms.jsonl", max_records=grpo_max_records)

    print("ES columns:", es_df.columns.tolist())
    print("GRPO columns:", grpo_df.columns.tolist())
    print("ES head:\n", es_df.head())
    print("GRPO head:\n", grpo_df.head())

    return es_df, grpo_df


def run_es_analysis(es_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ES random-walk / slope analysis and return the enriched reference DataFrame.
    """
    # Build trajectories and fit slopes
    param_delta_dict = build_param_delta_trajectories(es_df)
    param_slopes = fit_param_slopes(param_delta_dict)
    print("Example slopes:\n", param_slopes.head(10))

    # Use the same reference index as in the original code
    ref_param_info_slope = compute_ref_param_info_slope(es_df, param_slopes)
    print("Reference param info with slopes/head:\n", ref_param_info_slope.head())

    # Scatter plots
    plot_d_ratio_vs_r2(ref_param_info_slope)
    plot_numel_vs_d_ratio(ref_param_info_slope)

    # Heatmaps and extra diagnostics
    df_param_norms = add_layer_and_kind_columns(ref_param_info_slope)
    plot_param_heatmap(df_param_norms, value_col="r2", step=0, cmap="magma", vmax=1)
    plot_param_heatmap(df_param_norms, value_col="slope", step=0, cmap="magma")
    plot_param_heatmap(df_param_norms, value_col="d_ratio", step=0, cmap="coolwarm", vmin=0, vmax=1)

    # Large-numel subset, as in the original script
    big = df_param_norms.query("numel > 1e7")
    if not big.empty:
        plot_param_heatmap(big, value_col="r2", cmap="coolwarm", vmax=1, vmin=0.9990)
        plot_param_heatmap(big, value_col="1-d_ratio", cmap="coolwarm")

    # Kind-colored scatter plots
    plot_kind_colored_scatter(ref_param_info_slope)
    plot_scale_vs_linearity(ref_param_info_slope)

    return ref_param_info_slope


def main() -> None:
    """End-to-end ES vs GRPO param-norms analysis."""
    # Quick structural sanity check
    preview_raw_records()

    # Load ES/GRPO samples
    es_df, grpo_df = load_es_grpo_samples(es_max_records=10_000, grpo_max_records=10_000)

    # ES random-walk / slope analysis
    ref_param_info_slope = run_es_analysis(es_df)
    print("Final ES reference param info (with slopes):")
    print(ref_param_info_slope.head())

    # GRPO comparison heatmaps
    run_grpo_comparison(grpo_df)


if __name__ == "__main__":
    main()