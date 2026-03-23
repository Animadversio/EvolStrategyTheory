#!/usr/bin/env python3
"""Reproduce all figures from saved CSV/pkl data — no simulation needed.

Usage:
    python reproduce_figures.py                # reproduce all
    python reproduce_figures.py --exp 1        # only exp 1
    python reproduce_figures.py --exp D        # only exp D

All figures saved to STORE/figures/ as both PNG and PDF.
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ─────────────────────────────────────────────────────────────────────
STORE  = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2")
TABLES = STORE / "tables"
DATA   = STORE / "data"
FIGS   = STORE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "pdf.fonttype":    42,   # editable text in Illustrator/Inkscape
    "ps.fonttype":     42,
})
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
SIGMA_VALS = [0.05, 0.1, 0.2, 0.5, 1.0]
SIGMA_TAGS = ["sigma0_05", "sigma0_1", "sigma0_2", "sigma0_5", "sigma1_0"]


def savefig(fig, name):
    for ext in ["png", "pdf"]:
        path = FIGS / f"{name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {name}.png/.pdf")
    plt.close(fig)


def load_csv(name):
    p = TABLES / f"{name}.csv"
    if not p.exists():
        print(f"  [missing] {p.name}")
        return None
    return pd.read_csv(p)


def load_pkl(prefix, tag):
    p = DATA / f"{prefix}_{tag}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Flat landscape
# ══════════════════════════════════════════════════════════════════════════════
def reproduce_exp1():
    print("\n── Exp 1: Flat landscape ──")

    # 1a: drift vs N (one curve per d)
    df = load_csv("exp1a_drift_vs_Nd")
    if df is not None:
        D_VALS = sorted(df["d"].unique())
        fig, axes = plt.subplots(1, len(D_VALS), figsize=(4 * len(D_VALS), 4), sharey=False)
        if len(D_VALS) == 1:
            axes = [axes]
        for ai, d in enumerate(D_VALS):
            sub = df[df["d"] == d].sort_values("N")
            axes[ai].plot(sub["N"], sub["theory"], "k--", lw=1.5, label="Theory σ²d/4N")
            axes[ai].scatter(sub["N"], sub["emp"], color=COLORS[ai], s=40, zorder=5, label="Empirical")
            axes[ai].set_xscale("log"); axes[ai].set_yscale("log")
            axes[ai].set_xlabel("N"); axes[ai].set_ylabel("E[‖Δθ‖²]")
            axes[ai].set_title(f"d = {d}"); axes[ai].legend(); axes[ai].grid(True, alpha=0.3)
        plt.suptitle("Exp 1a: One-step drift variance vs N", fontsize=13)
        plt.tight_layout()
        savefig(fig, "exp1a_drift_vs_N")

        # 1a ratio heatmap (seaborn annotated)
        try:
            import seaborn as sns
            pivot = df.pivot(index="d", columns="N", values="ratio")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                        center=1.0, vmin=0.8, vmax=1.2, ax=ax,
                        linewidths=0.5, cbar_kws={"label": "emp / theory"})
            ax.set_title("Exp 1a: Ratio emp/theory (should = 1.0)")
            ax.set_xlabel("N"); ax.set_ylabel("d")
            plt.tight_layout()
            savefig(fig, "exp1a_ratio_heatmap")
        except ImportError:
            print("  [seaborn not available, skipping heatmap]")

    # 1b: sigma scaling
    df = load_csv("exp1b_sigma_scaling")
    if df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, (ax, xcol) in enumerate(zip(axes, ["sigma", "sigma"])):
            sub = df.sort_values("sigma")
            ax.plot(sub["sigma"], sub["theory"], "k--", lw=1.5, label="Theory")
            ax.scatter(sub["sigma"], sub["emp"], color=COLORS[0], s=50, zorder=5, label="Empirical")
            if i == 1:
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title("Log-log scale")
            else:
                ax.set_title("Linear scale")
            ax.set_xlabel("σ"); ax.set_ylabel("E[‖Δθ‖²]"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.suptitle("Exp 1b: σ scaling of per-step drift", fontsize=13)
        plt.tight_layout()
        savefig(fig, "exp1b_sigma_scaling")

    # 1c: cumulative drift
    df = load_csv("exp1c_cumulative")
    if df is not None:
        D_VALS = sorted(df["d"].unique())
        N_VALS = sorted(df["N"].unique())
        fig, axes = plt.subplots(1, len(D_VALS), figsize=(5 * len(D_VALS), 4))
        if len(D_VALS) == 1:
            axes = [axes]
        for ai, d in enumerate(D_VALS):
            for ni, N in enumerate(N_VALS):
                sub = df[(df["d"] == d) & (df["N"] == N)].sort_values("T")
                axes[ai].plot(sub["T"], sub["theory"], "--", color=COLORS[ni], lw=1.5)
                axes[ai].scatter(sub["T"], sub["emp"], color=COLORS[ni], s=20, label=f"N={N}")
            axes[ai].set_xlabel("T"); axes[ai].set_ylabel("E[‖θ_T−θ_0‖²]")
            axes[ai].set_title(f"d = {d}"); axes[ai].legend(fontsize=7); axes[ai].grid(True, alpha=0.3)
        plt.suptitle("Exp 1c: Cumulative drift vs T (random walk)", fontsize=13)
        plt.tight_layout()
        savefig(fig, "exp1c_cumulative_drift")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Linear landscape
# ══════════════════════════════════════════════════════════════════════════════
def reproduce_exp2():
    print("\n── Exp 2: Linear landscape ──")

    for fname, xcol, xlabel, title, figname in [
        ("exp2a_rho_vs_xi",  "xi",  "ξ (reward noise)", "Exp 2a: ρ vs reward noise ξ",    "exp2a_rho_vs_xi"),
        ("exp2b_rho_vs_N",   "N",   "N (population)",   "Exp 2b: ρ vs population N",       "exp2b_rho_vs_N"),
        ("exp2c_rho_vs_d",   "d",   "d (dimension)",    "Exp 2c: ρ vs dimension d",        "exp2c_rho_vs_d"),
    ]:
        df = load_csv(fname)
        if df is None:
            continue
        groups = df["d"].unique() if "d" in df.columns and xcol != "d" else [None]
        fig, ax = plt.subplots(figsize=(6, 4))
        for gi, grp in enumerate(sorted(groups)):
            sub = df if grp is None else df[df["d"] == grp]
            sub = sub.sort_values(xcol)
            label = f"d={grp}" if grp is not None else ""
            ax.plot(sub[xcol], sub["rho_theory"], "--", color=COLORS[gi], lw=1.5, label=f"Theory {label}")
            ax.scatter(sub[xcol], sub["rho_emp"], color=COLORS[gi], s=40, zorder=5, label=f"Emp {label}")
        if xcol in ("xi", "N", "d"):
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("ρ (on-manifold fraction)")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, figname)

    # 2e multistep — columns: N, xi, T, mean_dist
    df = load_csv("exp2e_multistep")
    if df is not None:
        step_col = "T" if "T" in df.columns else "step"
        dist_col = "mean_dist" if "mean_dist" in df.columns else "emp_dist"
        N_VALS = sorted(df["N"].unique()) if "N" in df.columns else [None]
        xi_vals = sorted(df["xi"].unique()) if "xi" in df.columns else [None]
        fig, ax = plt.subplots(figsize=(7, 4))
        for ni, N in enumerate(N_VALS):
            sub = df[df["N"] == N] if N is not None else df
            # pivot over xi if multiple xi values
            for xii, xi in enumerate(xi_vals[:3]):
                s = sub[sub["xi"] == xi].sort_values(step_col) if xi is not None else sub.sort_values(step_col)
                label = f"N={N}, ξ={xi}" if xi is not None else f"N={N}"
                ax.plot(s[step_col], s[dist_col], color=COLORS[ni], lw=2, alpha=0.7 - 0.2*xii, label=label)
        ax.set_xlabel("Step T"); ax.set_ylabel("‖θ_T − θ*‖")
        ax.set_title("Exp 2e: Multi-step convergence (linear)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, "exp2e_multistep_linear")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Quadratic landscape
# ══════════════════════════════════════════════════════════════════════════════
def reproduce_exp3():
    print("\n── Exp 3: Quadratic landscape ──")

    # 3a: sigma_R  (cols: spectrum, xi, sigmaR_th, sigmaR_emp, ...)
    df = load_csv("exp3a_sigmaR")
    if df is not None:
        th_col  = "sigmaR_th"  if "sigmaR_th"     in df.columns else "sigmaR_theory"
        emp_col = "sigmaR_emp" if "sigmaR_emp"     in df.columns else "sigmaR_empirical"
        fig, ax = plt.subplots(figsize=(6, 5))
        for gi, grp in enumerate(sorted(df["spectrum"].unique())):
            sub = df[df["spectrum"] == grp]
            ax.scatter(sub[th_col], sub[emp_col], color=COLORS[gi], s=50, label=grp)
        lims = [df[[th_col, emp_col]].min().min() * 0.9, df[[th_col, emp_col]].max().max() * 1.1]
        ax.plot(lims, lims, "k--", lw=1.5, label="y=x")
        ax.set_xlabel("σ_R theory"); ax.set_ylabel("σ_R empirical")
        ax.set_title("Exp 3a: σ_R theory vs empirical"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, "exp3a_sigmaR")

    # 3c: multistep  (cols: d, N, xi, T, dist, sigmaR_th, eff_step)
    df = load_csv("exp3c_multistep")
    if df is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        group_keys = [(d, N, xi) for d in sorted(df["d"].unique())[:3]
                                  for N in sorted(df["N"].unique())[:2]
                                  for xi in sorted(df["xi"].unique())[:2]]
        for gi, (d, N, xi) in enumerate(group_keys[:8]):
            sub = df[(df["d"]==d)&(df["N"]==N)&(df["xi"]==xi)].sort_values("T")
            ax.plot(sub["T"], sub["dist"], color=COLORS[gi%10], lw=1.5, label=f"d={d},N={N},ξ={xi}")
        ax.set_xlabel("Step T"); ax.set_ylabel("‖θ_T−θ*‖")
        ax.set_title("Exp 3c: Multi-step convergence"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, "exp3c_multistep_quad")

    # 3d: d scaling  (cols: d, T, dist, sigmaR_th, eff_step)
    df = load_csv("exp3d_d_scaling")
    if df is not None:
        T_final = df["T"].max()
        sub = df[df["T"] == T_final].sort_values("d")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub["d"], sub["dist"], color=COLORS[0], s=50, zorder=5, label="Empirical")
        ax.set_xscale("log"); ax.set_xlabel("d (dimension)"); ax.set_ylabel(f"‖θ_T−θ*‖ at T={T_final}")
        ax.set_title("Exp 3d: Final distance vs d"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, "exp3d_d_scaling")

    # 3e: N scaling  (cols: N, T, dist, ...)
    df = load_csv("exp3e_N_scaling")
    if df is not None:
        T_final = df["T"].max()
        sub = df[df["T"] == T_final].sort_values("N") if "T" in df.columns else df.sort_values("N")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub["N"], sub["dist"] if "dist" in sub.columns else sub.iloc[:,1],
                   color=COLORS[1], s=50, zorder=5, label="Empirical")
        ax.set_xscale("log"); ax.set_xlabel("N (population)"); ax.set_ylabel(f"‖θ_T−θ*‖ at T={T_final}")
        ax.set_title("Exp 3e: Final distance vs N"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, "exp3e_N_scaling")


# ══════════════════════════════════════════════════════════════════════════════
# EXP D — Multi-step GD comparison
# ══════════════════════════════════════════════════════════════════════════════
def reproduce_expD():
    print("\n── Exp D: Multi-step GD comparison ──")

    # D1: mean trajectories per sigma
    for sigma, tag in zip(SIGMA_VALS, SIGMA_TAGS):
        df = load_csv(f"expD1_trajectory_{tag}")
        if df is None:
            continue
        emp_cols = [c for c in df.columns if c.startswith("emp_mean_mode")]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # left: per-mode trajectories
        for i, col in enumerate(emp_cols[:10]):
            axes[0].plot(df["step"], df[col], color=COLORS[i % 10], lw=1, alpha=0.8)
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("Mode coordinate")
        axes[0].set_title(f"D1: Per-mode trajectories (σ={sigma})")
        axes[0].grid(True, alpha=0.3)
        # right: norm
        if "emp_norm" in df.columns:
            axes[1].plot(df["step"], df["emp_norm"], lw=2, label="Empirical ‖θ‖")
        if "theory_norm_frozen" in df.columns:
            axes[1].plot(df["step"], df["theory_norm_frozen"], "k--", lw=1.5, label="Theory (frozen γ)")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("‖θ_t‖")
        axes[1].set_title(f"D1: Distance from optimum (σ={sigma})")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.suptitle(f"Exp D1 — σ={sigma}", fontsize=12)
        plt.tight_layout()
        savefig(fig, f"expD1_mean_trajectory_{tag}")

    # D2: eigenmode decay rates (scatter theory vs emp)
    df_all = load_csv("expD2_decay_rates_all")
    if df_all is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, (sigma, sub) in enumerate(df_all.groupby("sigma")):
            sub = sub.dropna()
            ax.scatter(sub["theory_decay_rate"], sub["emp_decay_rate"],
                       color=COLORS[i], s=25, alpha=0.7, label=f"σ={sigma}")
        lim_vals = df_all[["theory_decay_rate","emp_decay_rate"]].replace([np.inf,-np.inf], np.nan).dropna()
        if len(lim_vals):
            lo = lim_vals.min().min() * 0.9; hi = lim_vals.max().max() * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y=x")
        ax.set_xlabel("Theory decay rate γ·λ"); ax.set_ylabel("Empirical decay rate")
        ax.set_title("Exp D2: Per-eigenmode decay rates"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig(fig, "expD2_decay_rates_summary")

    # D3: noise floor time series
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_summary = load_csv("expD3_noise_floor_summary")
    for i, (sigma, tag) in enumerate(zip(SIGMA_VALS, SIGMA_TAGS)):
        df_ts = load_csv(f"expD3_norm2_timeseries_{tag}")
        if df_ts is not None:
            axes[0].plot(df_ts["step"], df_ts["emp_norm2"], color=COLORS[i], lw=1.5, label=f"σ={sigma}")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("‖θ_t‖²")
    axes[0].set_title("D3: ‖θ‖² over time (each σ)"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    if df_summary is not None:
        axes[1].plot(df_summary["sigma"], df_summary["noise_floor_theory"], "k--", lw=2, label="Theory floor")
        axes[1].scatter(df_summary["sigma"], df_summary["stationary_var_emp_total"],
                        color=COLORS[0], s=60, zorder=5, label="Empirical (stationary)")
        axes[1].set_xscale("log"); axes[1].set_xlabel("σ"); axes[1].set_ylabel("Total stationary var")
        axes[1].set_title("D3: Noise floor theory vs emp"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle("Exp D3: Stationary noise floor", fontsize=12)
    plt.tight_layout()
    savefig(fig, "expD3_noise_floor_summary")

    # D4: ES vs GD per step and per function eval
    df_all = load_csv("expD4_es_vs_gd_all")
    if df_all is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        # Use first sigma for GD (same landscape)
        gd_ref = df_all[df_all["sigma"] == SIGMA_VALS[0]].sort_values("step")
        axes[0].plot(gd_ref["step"],  gd_ref["gd_norm_opt"],   "k--",  lw=2, label="GD (opt lr)")
        axes[0].plot(gd_ref["step"],  gd_ref["gd_norm_half"],  "k:",   lw=1.5, label="GD (½ lr)")
        axes[0].plot(gd_ref["step"],  gd_ref["gd_norm_noisy"], "k-.",  lw=1.5, label="GD (noisy)")
        axes[1].plot(gd_ref["fe_gd"], gd_ref["gd_norm_opt"],   "k--",  lw=2, label="GD (opt lr)")
        for i, (sigma, sub) in enumerate(df_all.groupby("sigma")):
            sub = sub.sort_values("step")
            axes[0].plot(sub["step"],  sub["es_norm"], color=COLORS[i], lw=2, label=f"ES σ={sigma}")
            axes[1].plot(sub["fe_es"], sub["es_norm"], color=COLORS[i], lw=2, label=f"ES σ={sigma}")
        for ax, xlabel in zip(axes, ["Step", "Function evaluations"]):
            ax.set_xlabel(xlabel); ax.set_ylabel("‖θ_t − θ*‖")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        axes[0].set_title("D4: Convergence per step")
        axes[1].set_title("D4: Convergence per function eval")
        plt.suptitle("Exp D4: ZScoreES vs GD", fontsize=12)
        plt.tight_layout()
        savefig(fig, "expD4_es_vs_gd_summary")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", help="Which exp to reproduce: 1, 2, 3, D, or all")
    args = parser.parse_args()

    funcs = {"1": reproduce_exp1, "2": reproduce_exp2, "3": reproduce_exp3, "D": reproduce_expD}
    if args.exp == "all":
        for fn in funcs.values():
            fn()
    elif args.exp in funcs:
        funcs[args.exp]()
    else:
        print(f"Unknown exp '{args.exp}'. Choose from: {list(funcs.keys())} or 'all'")
        sys.exit(1)

    print(f"\nAll figures saved to {FIGS}")
