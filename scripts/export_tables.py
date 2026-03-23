#!/usr/bin/env python3
"""Export all figure source data to tables/ as clean CSVs.

Covers:
  - Exp 1-3: copy existing CSVs + extract extra tables from pkl
  - Exp D1-D4: export per-figure tidy CSVs from pkl files
"""
import pickle, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

STORE = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2")
DATA   = STORE / "data"
TABLES = STORE / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SIGMA_VALS = [0.05, 0.1, 0.2, 0.5, 1.0]
SIGMA_TAGS = ["sigma0_05", "sigma0_1", "sigma0_2", "sigma0_5", "sigma1_0"]

def load(prefix, tag):
    p = DATA / f"{prefix}_{tag}.pkl"
    if not p.exists():
        print(f"  [missing] {p.name}")
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

# ── Copy existing Exp1-3 CSVs ─────────────────────────────────────────────────
print("Copying Exp 1-3 CSVs...")
for csv in DATA.glob("exp[123]*.csv"):
    shutil.copy(csv, TABLES / csv.name)
    print(f"  {csv.name}")

# ── Exp D1: mean trajectory (theory vs emp per sigma) ─────────────────────────
print("\nExporting Exp D1 tables...")
for sigma, tag in zip(SIGMA_VALS, SIGMA_TAGS):
    d = load("expD1_mean_trajectory", tag)
    if d is None:
        continue
    t_ax = np.array(d["t_ax"])
    rows = {"step": t_ax}
    # per-active-mode trajectories
    for key in ["emp_mean", "theory_frozen", "theory_dynamic"]:
        if key in d:
            arr = np.array(d[key])
            if arr.ndim == 2:
                # shape (T, n_modes) — save first 10 modes + mean
                for i in range(min(10, arr.shape[1])):
                    rows[f"{key}_mode{i}"] = arr[:, i]
            else:
                rows[key] = arr
    # also save norm (distance from origin)
    for key in ["emp_norm", "theory_norm_frozen"]:
        if key in d:
            rows[key] = np.array(d[key])
    df = pd.DataFrame(rows)
    fname = f"expD1_trajectory_{tag}.csv"
    df.to_csv(TABLES / fname, index=False)
    print(f"  {fname}  ({len(df)} rows, {len(df.columns)} cols)")

# ── Exp D2: per-eigenmode decay rates ────────────────────────────────────────
print("\nExporting Exp D2 tables...")
all_d2 = []
for sigma, tag in zip(SIGMA_VALS, SIGMA_TAGS):
    d = load("expD2_eigenmode_decay", tag)
    if d is None:
        continue
    th   = np.array(d["theory_decay_rates"])
    emp  = np.array(d["emp_rates"])
    lam  = np.array(d["lam"])[:len(th)]
    n    = min(len(th), len(emp))
    df = pd.DataFrame({
        "sigma":      sigma,
        "mode_idx":   np.arange(n),
        "eigenvalue": lam[:n],
        "theory_decay_rate": th[:n],
        "emp_decay_rate":    emp[:n],
        "ratio_emp_theory":  emp[:n] / np.clip(th[:n], 1e-12, None),
    })
    fname = f"expD2_decay_rates_{tag}.csv"
    df.to_csv(TABLES / fname, index=False)
    print(f"  {fname}  ({n} modes)")
    all_d2.append(df)
# combined
if all_d2:
    pd.concat(all_d2).to_csv(TABLES / "expD2_decay_rates_all.csv", index=False)
    print("  expD2_decay_rates_all.csv")

# ── Exp D3: noise floor / stationary variance ─────────────────────────────────
print("\nExporting Exp D3 tables...")
summary_d3 = []
for sigma, tag in zip(SIGMA_VALS, SIGMA_TAGS):
    d = load("expD3_noise_floor", tag)
    if d is None:
        continue
    t_ax = np.array(d["t_ax"])
    # time-series of ||θ||²
    df_ts = pd.DataFrame({
        "step":          t_ax,
        "emp_norm2":     np.array(d["emp_norm2"]),
    })
    df_ts.to_csv(TABLES / f"expD3_norm2_timeseries_{tag}.csv", index=False)
    print(f"  expD3_norm2_timeseries_{tag}.csv")
    # per-dimension stationary variance (downsample to avoid huge files)
    if "stationary_var_emp" in d and "stat_var_theory" in d:
        lam  = np.array(d["lam"])
        v_emp = np.array(d["stationary_var_emp"])
        v_th  = np.array(d["stat_var_theory"])
        df_dim = pd.DataFrame({
            "dim_idx":    np.arange(len(lam)),
            "eigenvalue": lam,
            "stat_var_emp":    v_emp,
            "stat_var_theory": v_th,
            "ratio":           v_emp / np.clip(v_th, 1e-12, None),
        })
        df_dim.to_csv(TABLES / f"expD3_stationary_var_{tag}.csv", index=False)
        print(f"  expD3_stationary_var_{tag}.csv")
    summary_d3.append({
        "sigma": sigma,
        "noise_floor_theory": float(d["noise_floor_total"]),
        "sigma_R_star":       float(d.get("sigma_R_star", np.nan)),
        "gamma_star":         float(d.get("gamma_star",   np.nan)),
        "stationary_var_emp_total": float(np.array(d["stationary_var_emp"]).sum()) if "stationary_var_emp" in d else np.nan,
    })
if summary_d3:
    pd.DataFrame(summary_d3).to_csv(TABLES / "expD3_noise_floor_summary.csv", index=False)
    print("  expD3_noise_floor_summary.csv")

# ── Exp D4: ES vs GD convergence ─────────────────────────────────────────────
print("\nExporting Exp D4 tables...")
all_d4 = []
for sigma, tag in zip(SIGMA_VALS, SIGMA_TAGS):
    d = load("expD4_es_vs_gd", tag)
    if d is None:
        continue
    t_ax = np.array(d["t_ax"])
    T = len(t_ax)
    df = pd.DataFrame({
        "sigma":         sigma,
        "step":          t_ax,
        "es_norm":       np.array(d["emp_norm"])[:T],
        "gd_norm_opt":   np.array(d["gd_norm_opt"])[:T],
        "gd_norm_half":  np.array(d["gd_norm_half"])[:T],
        "gd_norm_noisy": np.array(d["gd_norm_noisy"])[:T],
        "fe_es":         np.array(d["fe_es"])[:T],
        "fe_gd":         np.array(d["fe_gd"])[:T],
    })
    fname = f"expD4_es_vs_gd_{tag}.csv"
    df.to_csv(TABLES / fname, index=False)
    print(f"  {fname}  ({T} steps)")
    all_d4.append(df)
if all_d4:
    pd.concat(all_d4).to_csv(TABLES / "expD4_es_vs_gd_all.csv", index=False)
    print("  expD4_es_vs_gd_all.csv")

# ── Summary of all table files ─────────────────────────────────────────────────
tables = sorted(TABLES.glob("*.csv"))
print(f"\n{'='*60}")
print(f"Total: {len(tables)} CSV files in {TABLES}")
for f in tables:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<55} {size_kb:6.1f} KB")
