#!/usr/bin/env python3
"""Exp E: OU Variance Trajectory — Dynamic vs Frozen σ_R Theory.

PURPOSE
-------
The proposition in theory.tex gives the exact variance trajectory for a
**frozen-σ_R** OU process:

    Var[ci(t)] = (α²/N) · (1 − γk^{2t}) / (1 − γk²)
    γk = 1 − (ασ/σR) λk,   σR fixed at initial value σR(θ₀)

But σ_R is not fixed — it evolves as θ_t changes:
    σR(θ)² = σ²‖Qθ‖² + ½σ⁴ Tr[Q²] + ξ²

This experiment systematically tests:

  E1: Full variance trajectory Var[ci(t)] vs t — all three models
      Frozen σR(θ₀):  Var_i(t) = (α²/N) · (1−γ₀^{2t}) / (1−γ₀²)
      Dynamic (data):  Var_i(t+1) = γt² Var_i(t) + α²/N   [empirical σR(t)]
      Empirical:       variance across trials at each step

  E2: σR(t) trajectory — empirical vs analytic approximations
      σR(θ₀)       fixed at initial point
      σR(E[θ_t])   evaluated at mean trajectory (frozen-γ prediction)
      σR(√E[‖θt‖²]) evaluated at running RMS distance

  E3: σ sweep — how the frozen/dynamic gap varies with σ ∈ {0.01,0.05,0.1,0.2,0.5,1.0}
      At small σ: σR ≈ const → frozen model should work
      At large σ: σR changes a lot → frozen model breaks

  E4: Self-consistent σR — find σR* = σR(‖θ∞‖) at stationary distance
      and test if the frozen model with σR* instead of σR(θ₀) fixes the floor.
      The stationary norm satisfies: ‖θ∞‖² = Σi Var_i(∞) where Var_i depends on σR*.
      Solve this fixed-point equation numerically and validate.

─────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────

# Default local run:
  python scripts/exp_E_ou_variance_trajectory.py

# Custom:
  python scripts/exp_E_ou_variance_trajectory.py \\
      --d 500 --k 20 --N 50 --sigma 0.1 --T 600 --n_trials 300 \\
      --spectrum powerlaw --beta 1.0 --lam_max 5.0 --lam_min 0.1 \\
      --out_dir /path/to/output --tag run1

# Cluster sigma sweep:
  sbatch --array=0-5 scripts/sbatch/run_exp_E.sh

─────────────────────────────────────────────────────
Arguments
─────────────────────────────────────────────────────

Landscape:
  --d         INT    Dimension                         [500]
  --k         INT    Active (curved) eigenmodes        [20]
  --lam_max   FLOAT  Largest eigenvalue                [5.0]
  --lam_min   FLOAT  Smallest active eigenvalue        [0.1]
  --flat      FLOAT  Flat eigenvalue                   [1e-4]
  --spectrum  STR    uniform|powerlaw|range            [powerlaw]
  --beta      FLOAT  Power-law exponent                [1.0]

ZScoreES:
  --N         INT    Population size                   [50]
  --sigma     FLOAT  Exploration std σ                 [0.1]
  --xi        FLOAT  Observation noise ξ               [0.0]
  --ddof      INT    Std denominator 0=N, 1=N-1        [0]

Simulation:
  --T         INT    Steps per trial                   [600]
  --n_trials  INT    Independent trajectories          [300]
  --theta0_norm FLOAT ‖θ₀‖                             [10.0]
  --seed      INT    Random seed                       [42]

E3 sigma sweep:
  --sigma_list STR   Comma-separated σ values for E3   [0.01,0.05,0.1,0.2,0.5,1.0]

Output:
  --out_dir   STR    Output base dir                   [auto]
  --tag       STR    Tag appended to filenames         [""]
  --exps      STR    Sub-experiments to run            [E1,E2,E3,E4]
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── CLI ────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d",          type=int,   default=500)
    p.add_argument("--k",          type=int,   default=20)
    p.add_argument("--lam_max",    type=float, default=5.0)
    p.add_argument("--lam_min",    type=float, default=0.1)
    p.add_argument("--flat",       type=float, default=1e-4)
    p.add_argument("--spectrum",   type=str,   default="powerlaw")
    p.add_argument("--beta",       type=float, default=1.0)
    p.add_argument("--N",          type=int,   default=50)
    p.add_argument("--sigma",      type=float, default=0.1)
    p.add_argument("--xi",         type=float, default=0.0)
    p.add_argument("--ddof",       type=int,   default=0)
    p.add_argument("--T",          type=int,   default=600)
    p.add_argument("--n_trials",   type=int,   default=300)
    p.add_argument("--theta0_norm",type=float, default=10.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--sigma_list", type=str,   default="0.01,0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--out_dir",    type=str,   default=None)
    p.add_argument("--tag",        type=str,   default="")
    p.add_argument("--exps",       type=str,   default="E1,E2,E3,E4")
    return p.parse_args()


# ── paths ──────────────────────────────────────────────────────────────────────

def setup_paths(args):
    if args.out_dir:
        base = Path(args.out_dir)
    else:
        cluster = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation")
        local   = Path.home() / "DL_Projects/EvolStrategyTheory_validation"
        base    = cluster if cluster.exists() else local
    figs = base / "figures"; data = base / "data"
    figs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return figs, data


# ── landscape ─────────────────────────────────────────────────────────────────

def make_rotation(d, seed, device):
    gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
    A = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R)); signs[signs == 0] = 1.0
    return (Q * signs.unsqueeze(0)).to(device)


def make_eigenvalues(d, k, lam_max, lam_min, flat, spectrum, beta):
    eigs = torch.full((d,), flat, dtype=torch.float64)
    if k == 0: return eigs
    if spectrum == "uniform":
        eigs[:k] = lam_max
    elif spectrum == "powerlaw":
        idx = torch.arange(1, k+1, dtype=torch.float64)
        eigs[:k] = (lam_max * idx**-beta).clamp(min=lam_min)
    elif spectrum == "range":
        eigs[:k] = torch.logspace(np.log10(lam_max), np.log10(lam_min), k, dtype=torch.float64)
    return eigs


def build_landscape(args, device):
    U   = make_rotation(args.d, args.seed, device).to(torch.float64)
    lam = make_eigenvalues(args.d, args.k, args.lam_max, args.lam_min,
                           args.flat, args.spectrum, args.beta).to(device)
    return U, lam


# ── σ_R formula ───────────────────────────────────────────────────────────────

def compute_sigma_R(theta, U, lam, sigma, xi):
    """σ_R(θ) = sqrt(σ²‖Qθ‖² + ½σ⁴Tr[Q²] + ξ²)"""
    v       = U @ (lam * (U.T @ theta))          # Qθ
    v_norm2 = (v**2).sum().item()
    tr_Q2   = (lam**4).sum().item()              # Tr[Q²] = Σλi²
    return float(np.sqrt(sigma**2 * v_norm2 + 0.5 * sigma**4 * tr_Q2 + xi**2 + 1e-30))


def sigma_R_at_norm(norm_theta, lam, sigma, xi, k):
    """
    Mean-field σ_R as function of scalar ‖θ‖.

    σ_R²(θ) = σ²‖Qθ‖² + ½σ⁴Tr[Q²] + ξ²

    If θ is isotropically distributed with ‖θ‖=r over all d dims:
      E[‖Qθ‖²] = (r²/d)·Tr[Q²]   where Tr[Q²] = Σλi²

    If θ is isotropically distributed only over the k active dims:
      E[‖Qθ‖²] = (r²/k)·Σ_{i<k} λi²

    We use the active-modes version (more accurate in practice).
    Tr[Q²] = Σλi²   (Q has eigenvalues λi, so Q² has eigenvalues λi²)
    """
    lam_np   = lam.cpu().numpy()
    lam_act  = lam_np[:k]                     # active eigenvalues only
    tr_Q2    = (lam_np**2).sum()              # Tr[Q²] = Σλi² (all dims)
    tr_Q2_act = (lam_act**2).sum()            # Σλi² for active dims only
    # Use active-dims mean-field: E[‖Qθ‖²] ≈ (r²/k)·Σ_{active} λi²
    v_norm2_mf = (norm_theta**2 / max(k, 1)) * tr_Q2_act
    return float(np.sqrt(sigma**2 * v_norm2_mf + 0.5 * sigma**4 * tr_Q2 + xi**2 + 1e-30))


# ── OU variance trajectory formulas ──────────────────────────────────────────

def ou_var_frozen(lam_np, sigma_R, sigma, alpha, N, T):
    """
    Frozen-σ_R OU variance trajectory (closed form from proposition):
      Var_i(t) = (α²/N) · (1 − γi^{2t}) / (1 − γi²)
    with γi = 1 − (ασ/σR) λi.
    Returns array (T+1, d).
    """
    gamma  = 1.0 - (alpha * sigma / sigma_R) * lam_np   # (d,)
    gamma2 = gamma**2
    t_ax   = np.arange(T+1)
    # Var_i(t) = (α²/N) · (1 − γi^{2t}) / (1 − γi²)
    # Handle γi²=1 case (flat dims) separately
    var_traj = np.zeros((T+1, len(lam_np)))
    stable   = np.abs(gamma) < 1.0
    for i in range(len(lam_np)):
        if stable[i] and abs(1.0 - gamma2[i]) > 1e-10:
            exponent = gamma2[i]**t_ax        # γi^{2t}
            var_traj[:, i] = (alpha**2 / N) * (1.0 - exponent) / (1.0 - gamma2[i])
        else:
            # flat / boundary: Var(t) = (α²/N)·t  (random walk)
            var_traj[:, i] = (alpha**2 / N) * t_ax
    return var_traj, gamma


def ou_var_dynamic(sigma_R_traj_mean, lam_np, sigma, alpha, N, T):
    """
    Dynamic-σ_R OU variance trajectory (numerical recursion):
      Var_i(t+1) = γt² · Var_i(t) + α²/N
    where γt = 1 − (ασ/σR(t)) λi and σR(t) = mean empirical σ_R at step t.
    Returns array (T+1, d).
    """
    var_i  = np.zeros(len(lam_np))
    var_traj = np.zeros((T+1, len(lam_np)))
    var_traj[0] = 0.0
    for t in range(T):
        gamma_t = 1.0 - (alpha * sigma / max(sigma_R_traj_mean[t], 1e-12)) * lam_np
        var_i   = gamma_t**2 * var_i + alpha**2 / N
        var_traj[t+1] = var_i
    return var_traj


def ou_var_self_consistent(lam_np, sigma, alpha, N, xi, d, tol=1e-8, max_iter=2000):
    """
    Self-consistent σ_R fixed-point (E4) — exact equation, no mean-field.

    σ_R*² = σ² · E[‖Qθ_∞‖²] + ½σ⁴Tr[Q²] + ξ²

    At stationarity, using the frozen-OU variance formula self-consistently:
      E[‖Qθ_∞‖²] = Σᵢ λᵢ² · Var[cᵢ(∞; σ_R*)]
    where Var[cᵢ(∞; σ_R*)] = (α²/N) / (1 − γᵢ²),  γᵢ = 1 − (ασ/σ_R*)λᵢ

    This gives the exact fixed-point equation in σ_R* alone:
      σ_R*² = σ² · Σᵢ λᵢ² (α²/N)/(1−γᵢ²(σ_R*)) + ½σ⁴Tr[Q²] + ξ²

    Iterate with damping for stability.
    Returns σR_star, gamma_star, stat_vars_star, convergence_history.
    """
    tr_Q2    = (lam_np**2).sum()          # Tr[Q²] = Σλᵢ²
    baseline = 0.5 * sigma**4 * tr_Q2 + xi**2

    def stat_var(sR):
        gamma  = 1.0 - (alpha * sigma / max(sR, 1e-12)) * lam_np
        gamma2 = gamma**2
        stable = np.abs(gamma) < 1.0 - 1e-9
        denom  = np.where(stable, 1.0 - gamma2, np.inf)
        return np.where(stable, (alpha**2 / N) / denom, np.inf)

    def rhs(sR):
        sv   = stat_var(sR)
        v_sq = (lam_np**2 * np.clip(sv, 0, 1e10)).sum()   # E[‖Qθ‖²] = Σλᵢ²·Varᵢ
        return float(np.sqrt(sigma**2 * v_sq + baseline + 1e-30))

    # Initialize at σ_R = σ_R at θ=0 (minimal value, only curvature term)
    sR = float(np.sqrt(baseline + 1e-30))
    history = [sR]
    rho = 0.2   # conservative damping

    for it in range(max_iter):
        sR_new = (1 - rho) * sR + rho * rhs(sR)
        history.append(sR_new)
        if abs(sR_new - sR) / max(sR, 1e-12) < tol:
            sR = sR_new
            break
        sR = sR_new

    sv    = stat_var(sR)
    gamma = 1.0 - (alpha * sigma / max(sR, 1e-12)) * lam_np
    return sR, gamma, sv, np.array(history)


# ── ZScoreES step ─────────────────────────────────────────────────────────────

def zscore_es_step(theta, U, lam, sigma, alpha, N, xi, ddof, device):
    d   = theta.shape[0]
    eps = torch.randn(N, d, device=device, dtype=torch.float64)
    v   = U @ (lam * (U.T @ theta))                              # Qθ
    Qeps = (U @ (lam.unsqueeze(-1) * (U.T @ eps.T))).T
    R   = -sigma*(eps@v) - 0.5*sigma**2*(eps*Qeps).sum(-1)
    if xi > 0: R = R + xi*torch.randn(N, device=device, dtype=torch.float64)
    mu_R    = R.mean()
    sigma_R = R.std(correction=ddof).clamp(min=1e-12)
    Z       = (R - mu_R) / sigma_R
    delta   = alpha / N * (Z.unsqueeze(-1) * eps).sum(0)
    return theta + delta, sigma_R.item()


# ── simulation ────────────────────────────────────────────────────────────────

def run_trials(theta0, U, lam, args, device):
    """
    Returns:
      traj_coords  (n_trials, T+1, d)  — eigenbasis coords ci(t) = ui^T θ_t
      sigma_R_traj (n_trials, T)       — empirical σ_R at each step
      norm_traj    (n_trials, T+1)     — ‖θ_t‖ at each step
    """
    n, T, d = args.n_trials, args.T, args.d
    alpha   = args.sigma / 2.0
    UT      = U.T
    traj_coords  = np.zeros((n, T+1, d), dtype=np.float32)
    sigma_R_traj = np.zeros((n, T),      dtype=np.float32)
    norm_traj    = np.zeros((n, T+1),    dtype=np.float32)

    for trial in range(n):
        theta = theta0.clone()
        traj_coords[trial, 0]  = (UT @ theta).cpu().numpy()
        norm_traj[trial, 0]    = theta.norm().item()
        for t in range(T):
            theta, sR = zscore_es_step(theta, U, lam, args.sigma, alpha,
                                        args.N, args.xi, args.ddof, device)
            traj_coords[trial, t+1]  = (UT @ theta).cpu().numpy()
            sigma_R_traj[trial, t]   = sR
            norm_traj[trial, t+1]    = theta.norm().item()
        if (trial+1) % 50 == 0:
            print(f"  trial {trial+1}/{n}", flush=True)

    return traj_coords, sigma_R_traj, norm_traj


def savefig(fig, path, tag=""):
    stem = path.stem + (f"_{tag}" if tag else "")
    out  = path.parent / (stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)
    return out


COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ══════════════════════════════════════════════════════════════════════════════
# E1: Full variance trajectory Var[ci(t)] vs t
# ══════════════════════════════════════════════════════════════════════════════

def exp_E1(traj_coords, sigma_R_traj, norm_traj, theta0, U, lam, args, figs, data, tag):
    print("\n[E1] Full variance trajectory Var[ci(t)] vs t", flush=True)
    T, d = args.T, args.d
    alpha = args.sigma / 2.0
    lam_np = lam.cpu().numpy()
    t_ax   = np.arange(T+1)

    # Empirical variance across trials at each step
    emp_var = traj_coords.var(axis=0)           # (T+1, d)

    # Frozen σ_R at θ₀
    sigma_R0 = compute_sigma_R(theta0, U, lam, args.sigma, args.xi)
    var_frozen, gamma0 = ou_var_frozen(lam_np, sigma_R0, args.sigma, alpha, args.N, T)

    # Dynamic σ_R (empirical mean)
    mean_sR = sigma_R_traj.mean(axis=0)         # (T,)
    var_dynamic = ou_var_dynamic(mean_sR, lam_np, args.sigma, alpha, args.N, T)

    print(f"  σ_R(θ₀) = {sigma_R0:.4f},  γ_max = {gamma0[:args.k].max():.4f},  γ_min = {gamma0[:args.k].min():.4f}", flush=True)

    n_show = min(6, args.k)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, i in enumerate(range(n_show)):
        ax = axes[idx]
        ax.plot(t_ax, emp_var[:, i],      color="k",  lw=2,   label="Empirical")
        ax.plot(t_ax, var_frozen[:, i],   color="C0", lw=1.5, ls="--", label=f"Frozen σ_R={sigma_R0:.3f}")
        ax.plot(t_ax, var_dynamic[:, i],  color="C2", lw=1.5, ls=":",  label="Dynamic σ_R(t)")
        ax.set_title(f"Mode {i}: λ={lam_np[i]:.3f}, γ={gamma0[i]:.4f}")
        ax.set_xlabel("Step t"); ax.set_ylabel("Var[ci(t)]")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(f"E1: Var[ci(t)] trajectory — d={args.d} N={args.N} σ={args.sigma}")
    savefig(fig, figs/"expE1_var_trajectory.png", tag)

    # Ratio panels: frozen/empirical and dynamic/empirical over time
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    for i in range(n_show):
        c = COLORS[i]
        ratio_f = var_frozen[:, i] / (emp_var[:, i] + 1e-20)
        ratio_d = var_dynamic[:, i] / (emp_var[:, i] + 1e-20)
        axes2[0].plot(t_ax, ratio_f, color=c, lw=1.5, label=f"λ={lam_np[i]:.2f}")
        axes2[1].plot(t_ax, ratio_d, color=c, lw=1.5, label=f"λ={lam_np[i]:.2f}")
    for ax, title in zip(axes2, ["Frozen/Empirical", "Dynamic/Empirical"]):
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_ylim(0, 5); ax.set_xlabel("Step t"); ax.set_ylabel("Theory/Empirical")
        ax.set_title(f"E1 Ratio: {title}"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig2.suptitle(f"E1: Ratio theory/empirical — d={args.d} N={args.N} σ={args.sigma}")
    savefig(fig2, figs/"expE1_var_ratio.png", tag)

    res = {
        "emp_var": emp_var, "var_frozen": var_frozen, "var_dynamic": var_dynamic,
        "sigma_R0": sigma_R0, "gamma0": gamma0, "mean_sR": mean_sR,
        "t_ax": t_ax, "lam": lam_np, "args": vars(args),
    }
    pkl = data / f"expE1_var_trajectory{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# E2: σ_R(t) trajectory — empirical vs analytic approximations
# ══════════════════════════════════════════════════════════════════════════════

def exp_E2(traj_coords, sigma_R_traj, norm_traj, theta0, U, lam, args, figs, data, tag):
    print("\n[E2] σ_R(t) trajectory — empirical vs approximations", flush=True)
    T, d = args.T, args.d
    alpha = args.sigma / 2.0
    lam_np = lam.cpu().numpy()
    t_ax   = np.arange(T+1)

    # Empirical σ_R mean and std over trials
    mean_sR = sigma_R_traj.mean(axis=0)   # (T,)
    std_sR  = sigma_R_traj.std(axis=0)

    # σ_R(θ₀) — constant baseline
    sigma_R0 = compute_sigma_R(theta0, U, lam, args.sigma, args.xi)
    sR_frozen = np.full(T, sigma_R0)

    # σ_R evaluated at mean-field ‖θ_t‖: use running mean norm
    mean_norm = norm_traj.mean(axis=0)    # (T+1,)
    sR_meanfield = np.array([sigma_R_at_norm(mean_norm[t], lam, args.sigma, args.xi, args.k)
                              for t in range(T)])

    # σ_R evaluated at frozen-theory mean ‖E[θ_t]‖ (from frozen OU prediction)
    _, gamma0 = ou_var_frozen(lam_np, sigma_R0, args.sigma, alpha, args.N, T)
    emp_mean  = traj_coords.mean(axis=0)  # (T+1, d)
    emp_norm  = np.sqrt((emp_mean**2).sum(axis=-1))  # (T+1,)
    sR_theory_norm = np.array([sigma_R_at_norm(emp_norm[t], lam, args.sigma, args.xi, args.k)
                                for t in range(T)])

    print(f"  σ_R(θ₀) = {sigma_R0:.4f}", flush=True)
    print(f"  σ_R empirical at t=T: {mean_sR[-1]:.4f} ± {std_sR[-1]:.4f}", flush=True)
    print(f"  σ_R mean-field at t=T: {sR_meanfield[-1]:.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(range(T), mean_sR, color="k", lw=2, label="Empirical σ_R(t)")
    ax.fill_between(range(T), mean_sR-std_sR, mean_sR+std_sR, alpha=0.2, color="k")
    ax.axhline(sigma_R0, color="C0", ls="--", lw=1.5, label=f"Frozen σ_R(θ₀)={sigma_R0:.3f}")
    ax.plot(range(T), sR_meanfield, color="C1", ls="--", lw=1.5, label="Mean-field σ_R(‖E[θ]‖)")
    ax.plot(range(T), sR_theory_norm, color="C2", ls=":", lw=1.5, label="σ_R at frozen-theory norm")
    ax.set_xlabel("Step t"); ax.set_ylabel("σ_R(t)")
    ax.set_title("E2a: σ_R trajectory"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_ax, mean_norm, color="k", lw=2, label="Empirical E[‖θ_t‖]")
    # frozen-OU mean ‖θ‖: Σi (1-γi^{2t}) |ci(0)|² + ... but just use emp_norm
    ax.plot(t_ax, emp_norm, color="C0", ls="--", lw=1.5, label="|E[θ_t]| (mean of mean)")
    ax.set_xlabel("Step t"); ax.set_ylabel("‖θ_t‖")
    ax.set_title("E2b: ‖θ_t‖ trajectory"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"E2: σ_R trajectory — d={args.d} N={args.N} σ={args.sigma}")
    savefig(fig, figs/"expE2_sigmaR_trajectory.png", tag)

    res = {
        "mean_sR": mean_sR, "std_sR": std_sR,
        "sR_frozen": sR_frozen, "sR_meanfield": sR_meanfield,
        "sR_theory_norm": sR_theory_norm,
        "mean_norm": mean_norm, "emp_norm": emp_norm,
        "sigma_R0": sigma_R0, "lam": lam_np, "args": vars(args),
    }
    pkl = data / f"expE2_sigmaR_trajectory{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# E3: σ sweep — how frozen/dynamic gap varies with σ
# ══════════════════════════════════════════════════════════════════════════════

def exp_E3(theta0, U, lam, args, device, figs, data, tag):
    print("\n[E3] σ sweep — frozen vs dynamic gap vs σ", flush=True)
    sigma_list = [float(s) for s in args.sigma_list.split(",")]
    T, d = args.T, args.d
    lam_np = lam.cpu().numpy()

    summary = []
    for sigma in sigma_list:
        alpha = sigma / 2.0
        print(f"  σ={sigma:.3f} ...", flush=True)

        # Run short simulation
        n_short = min(args.n_trials, 150)
        args_s = type('A', (), vars(args).copy())()
        args_s.sigma = sigma; args_s.n_trials = n_short

        traj, sR_traj, norm_traj = run_trials(theta0, U, lam, args_s, device)

        # Empirical stationary variance (last 20%)
        stat_start = int(0.8 * T)
        emp_stat = traj[:, stat_start:, :].var(axis=(0,1))  # (d,)

        # Frozen model stationary variance
        sigma_R0 = compute_sigma_R(theta0, U, lam, sigma, args.xi)
        _, gamma0 = ou_var_frozen(lam_np, sigma_R0, sigma, alpha, args.N, T)
        denom0 = np.maximum(1.0 - gamma0**2, 1e-10)
        stat_frozen = (alpha**2 / args.N) / denom0   # (d,)

        # Dynamic model stationary variance (end of recursion)
        mean_sR = sR_traj.mean(axis=0)
        var_dyn = ou_var_dynamic(mean_sR, lam_np, sigma, alpha, args.N, T)
        stat_dynamic = var_dyn[-1]   # last step

        n_act = args.k
        ratio_frozen  = stat_frozen[:n_act].sum() / (emp_stat[:n_act].sum() + 1e-15)
        ratio_dynamic = stat_dynamic[:n_act].sum() / (emp_stat[:n_act].sum() + 1e-15)
        sR_ratio      = mean_sR[-1] / sigma_R0 if sigma_R0 > 1e-10 else np.nan

        summary.append({
            "sigma": sigma, "sigma_R0": sigma_R0,
            "sigma_R_final": mean_sR[-1],
            "sR_ratio": sR_ratio,
            "ratio_frozen": ratio_frozen,
            "ratio_dynamic": ratio_dynamic,
            "emp_stat_sum": emp_stat[:n_act].sum(),
            "frozen_stat_sum": stat_frozen[:n_act].sum(),
            "dynamic_stat_sum": stat_dynamic[:n_act].sum(),
        })
        print(f"    σ_R0={sigma_R0:.4f}  σ_R_final={mean_sR[-1]:.4f}  "
              f"ratio_frozen={ratio_frozen:.3f}  ratio_dynamic={ratio_dynamic:.3f}", flush=True)

    # Plot
    sigs    = [s["sigma"]          for s in summary]
    r_froz  = [s["ratio_frozen"]   for s in summary]
    r_dyn   = [s["ratio_dynamic"]  for s in summary]
    sR_rat  = [s["sR_ratio"]       for s in summary]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogx(sigs, r_froz, "C0o-", ms=7, lw=2, label="Frozen σ_R(θ₀) / Empirical")
    ax.semilogx(sigs, r_dyn,  "C2s-", ms=7, lw=2, label="Dynamic σ_R(t) / Empirical")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("σ (exploration std)"); ax.set_ylabel("Σ Var theory / Σ Var empirical")
    ax.set_title("E3a: Frozen vs Dynamic gap across σ")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogx(sigs, sR_rat, "C1^-", ms=7, lw=2)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("σ"); ax.set_ylabel("σ_R(T) / σ_R(θ₀)")
    ax.set_title("E3b: σ_R change ratio (final/initial)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"E3: σ sweep — d={args.d} N={args.N}")
    savefig(fig, figs/"expE3_sigma_sweep.png", tag)

    res = {"summary": summary, "args": vars(args)}
    pkl = data / f"expE3_sigma_sweep{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# E4: Self-consistent σ_R fixed-point
# ══════════════════════════════════════════════════════════════════════════════

def exp_E4(traj_coords, sigma_R_traj, theta0, U, lam, args, figs, data, tag):
    print("\n[E4] Self-consistent σ_R fixed-point", flush=True)
    T, d = args.T, args.d
    alpha = args.sigma / 2.0
    lam_np = lam.cpu().numpy()

    # Empirical stationary variance
    stat_emp = traj_coords[:, int(0.8*T):, :].var(axis=(0,1))   # (d,)

    # Three models
    sigma_R0 = compute_sigma_R(theta0, U, lam, args.sigma, args.xi)
    _, gamma0 = ou_var_frozen(lam_np, sigma_R0, args.sigma, alpha, args.N, T)
    denom0    = np.maximum(1.0 - gamma0**2, 1e-10)
    stat_frozen = (alpha**2 / args.N) / denom0

    sR_sc, gamma_sc, stat_sc, sc_history = ou_var_self_consistent(
        lam_np, args.sigma, alpha, args.N, args.xi, d)

    print(f"  σ_R(θ₀)        = {sigma_R0:.5f}", flush=True)
    print(f"  σ_R self-cons  = {sR_sc:.5f}  (converged in {len(sc_history)} iters)", flush=True)
    print(f"  σ_R empirical final = {sigma_R_traj.mean(axis=0)[-1]:.5f}", flush=True)
    print(f"  Σ Var: frozen={stat_frozen[:args.k].sum():.3f}  "
          f"self-cons={stat_sc[:args.k].sum():.3f}  emp={stat_emp[:args.k].sum():.3f}", flush=True)

    n_show = min(args.k+3, 30)
    mode_idx = np.arange(n_show)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.semilogy(mode_idx, stat_emp[mode_idx],    "ko-", ms=5, lw=2,   label="Empirical")
    ax.semilogy(mode_idx, stat_frozen[mode_idx], "C0x--", ms=5, lw=1.5, label=f"Frozen σ_R(θ₀)={sigma_R0:.3f}")
    ax.semilogy(mode_idx, stat_sc[mode_idx],     "C3s--", ms=5, lw=1.5, label=f"Self-cons σ_R*={sR_sc:.3f}")
    ax.set_xlabel("Mode i"); ax.set_ylabel("Var[ci(∞)]")
    ax.set_title("E4a: Per-mode stationary variance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    eps = 1e-15
    ax.plot(mode_idx, stat_frozen[mode_idx]/(stat_emp[mode_idx]+eps), "C0o-", ms=5, lw=1.5, label="Frozen/Emp")
    ax.plot(mode_idx, stat_sc[mode_idx]/(stat_emp[mode_idx]+eps),     "C3s-", ms=5, lw=1.5, label="Self-cons/Emp")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_ylim(0, max(5, stat_frozen[:n_show].max()/(stat_emp[:n_show].max()+eps)*1.1))
    ax.set_xlabel("Mode i"); ax.set_ylabel("Theory/Empirical")
    ax.set_title("E4b: Ratio theory/empirical"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(sc_history, "C3o-", ms=5, lw=1.5)
    ax.axhline(sigma_R0, color="C0", ls="--", lw=1.5, label=f"σ_R(θ₀)={sigma_R0:.3f}")
    ax.axhline(sigma_R_traj.mean(axis=0)[-1], color="k", ls=":", lw=1.5, label="Empirical σ_R final")
    ax.set_xlabel("Iteration"); ax.set_ylabel("σ_R")
    ax.set_title("E4c: Self-consistent fixed-point iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"E4: Self-consistent σ_R* — d={args.d} N={args.N} σ={args.sigma}")
    savefig(fig, figs/"expE4_self_consistent.png", tag)

    res = {
        "stat_emp": stat_emp, "stat_frozen": stat_frozen, "stat_sc": stat_sc,
        "sigma_R0": sigma_R0, "sR_sc": sR_sc, "gamma_sc": gamma_sc,
        "sc_history": sc_history, "lam": lam_np, "args": vars(args),
    }
    pkl = data / f"expE4_self_consistent{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Config: d={args.d} k={args.k} N={args.N} σ={args.sigma} ξ={args.xi} "
          f"T={args.T} n_trials={args.n_trials}", flush=True)

    figs, data = setup_paths(args)
    exps = [e.strip() for e in args.exps.split(",")]

    U, lam = build_landscape(args, device)
    torch.manual_seed(args.seed + 1)
    theta0 = torch.randn(args.d, dtype=torch.float64, device=device)
    theta0 = theta0 / theta0.norm() * args.theta0_norm

    sigma_R0 = compute_sigma_R(theta0, U, lam, args.sigma, args.xi)
    print(f"σ_R(θ₀) = {sigma_R0:.5f},  ‖θ₀‖ = {theta0.norm():.2f}", flush=True)

    # E3 uses its own inner simulation loop per σ, so we can skip the main sim for E3-only
    need_sim = any(e in exps for e in ["E1","E2","E4"])
    traj_coords = sigma_R_traj = norm_traj = None

    if need_sim:
        print(f"\nSimulating {args.n_trials} trials × {args.T} steps ...", flush=True)
        t0 = time.time()
        traj_coords, sigma_R_traj, norm_traj = run_trials(theta0, U, lam, args, device)
        print(f"Done in {time.time()-t0:.1f}s", flush=True)

        sim_pkl = data / f"expE_simulation{('_'+args.tag) if args.tag else ''}.pkl"
        with open(sim_pkl, "wb") as f:
            pickle.dump({"traj_coords": traj_coords, "sigma_R_traj": sigma_R_traj,
                         "norm_traj": norm_traj, "theta0": theta0.cpu().numpy(),
                         "lam": lam.cpu().numpy(), "args": vars(args)}, f)
        print(f"Simulation saved: {sim_pkl}", flush=True)

    if "E1" in exps:
        exp_E1(traj_coords, sigma_R_traj, norm_traj, theta0, U, lam, args, figs, data, args.tag)
    if "E2" in exps:
        exp_E2(traj_coords, sigma_R_traj, norm_traj, theta0, U, lam, args, figs, data, args.tag)
    if "E3" in exps:
        exp_E3(theta0, U, lam, args, device, figs, data, args.tag)
    if "E4" in exps:
        exp_E4(traj_coords, sigma_R_traj, theta0, U, lam, args, figs, data, args.tag)

    print(f"\nAll done. Figures → {figs}", flush=True)


if __name__ == "__main__":
    main()
