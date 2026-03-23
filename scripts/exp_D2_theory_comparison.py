#!/usr/bin/env python3
"""Exp D2: Multi-step ES — systematic theory comparison with three noise models.

PURPOSE
-------
The original exp_D used a noise floor formula that mixed two different
approximation levels. This script tests them side-by-side to identify
which correction terms matter and where the theory needs refinement.

THREE NOISE MODELS FOR THE STATIONARY VARIANCE (D3)
-----------------------------------------------------
The ES multi-step dynamic is:
  θ_{t+1} = (I - γQ) θ_t + η_t

The key question is: what is Cov[η_t]?

  Model 0 — Simplified theory (from new Proposition in theory.tex):
    η_t ~ N(0, α²/N · I_d)
    Noise is purely isotropic. Stationary variance per mode:
      Var[c_i(∞)] = (α²/N) / (1 - γ_k²)
    where γ_k = 1 - (ασ/σ_R)λ_k, σ_R frozen at θ*=0.

  Model 1 — Full Prop 3 Cov[Δθ] (isotropic + anisotropic Q² term):
    Cov[η_t] = (α²/N σ_R²)(σ² vv^T + 2σ⁴Q² + σ_R² I_d)
    Near θ*=0 (v→0): Cov[η_t] ≈ (α²/N)(I_d + 2σ⁴Q²/σ_R²)
    Per-mode noise: α²/N + 2σ⁴α²λ_i²/(N σ_R²)
    (This is what exp_D originally coded — but it's not what the proposition says.)

  Model 2 — Dynamic γ correction (γ grows as θ→0):
    Use the empirical σ_R trajectory instead of frozen γ*.
    Not a closed-form formula — compute numerically from simulation.

EXPERIMENTS
-----------
  D1: Mean trajectory E[θ_t] vs frozen-γ theory — same as exp_D
  D2: Per-eigenmode decay rates — same as exp_D
  D3: Noise floor comparison — ALL THREE MODELS vs empirical Var[c_i(∞)]
      This is the main new experiment: directly shows which model fits.
  D4: ES vs GD — same as exp_D

Outputs (all paths under --out_dir, default ~/DL_Projects/EvolStrategyTheory_validation/):
  data/expD_simulation[_tag].pkl   — raw traj_coords (n_trials × T+1 × d)
  data/expD1_mean_trajectory.pkl
  data/expD2_eigenmode_decay.pkl
  data/expD3_noise_floor.pkl
  data/expD4_es_vs_gd.pkl
  figures/expD{1-4}_*.png

─────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────

# Quick local test (CPU/GPU auto-detected):
  python scripts/exp_D_multistep_gd_comparison.py

# Full local run:
  python scripts/exp_D_multistep_gd_comparison.py \\
      --d 1000 --k 20 --N 50 --sigma 0.1 --xi 0.0 \\
      --T 500 --n_trials 200 --theta0_norm 10.0 \\
      --spectrum powerlaw --beta 1.0 \\
      --lam_max 5.0 --lam_min 0.1 --ddof 0

# Run only specific sub-experiments:
  python ... --exps D1,D3

# Custom output dir + tag (useful for parameter sweeps):
  python ... --out_dir /path/to/output --tag sigma05

# On cluster via sbatch (see scripts/sbatch/run_exp_D.sh):
  sbatch scripts/sbatch/run_exp_D.sh            # single run
  sbatch --array=0-4 scripts/sbatch/run_exp_D.sh  # sigma sweep

─────────────────────────────────────────────────────
Arguments
─────────────────────────────────────────────────────

Landscape:
  --d          INT    Parameter space dimension               [default: 500]
  --k          INT    Number of curved eigenmodes             [default: 20]
  --lam_max    FLOAT  Largest eigenvalue                      [default: 5.0]
  --lam_min    FLOAT  Smallest active eigenvalue              [default: 0.5]
  --flat       FLOAT  Eigenvalue for inactive dims            [default: 1e-4]
  --spectrum   STR    uniform | powerlaw | range              [default: powerlaw]
  --beta       FLOAT  Power-law exponent (spectrum=powerlaw)  [default: 1.0]

ZScoreES:
  --N          INT    Population size per step                [default: 50]
  --sigma      FLOAT  Exploration std σ                       [default: 0.1]
  --xi         FLOAT  Observation noise std ξ                 [default: 0.0]
  --ddof       INT    Std denominator: 0=N (theory), 1=N-1   [default: 0]

Simulation:
  --T          INT    Number of ES steps per trial            [default: 400]
  --n_trials   INT    Number of independent trajectories      [default: 200]
  --theta0_norm FLOAT ‖θ_0‖ starting distance from θ*        [default: 10.0]
  --seed       INT    Random seed                             [default: 42]

GD (D4):
  --gd_lr      FLOAT  GD learning rate (default: 2/(λ_min+λ_max))

Output:
  --out_dir    STR    Output directory                        [default: auto]
  --tag        STR    Tag appended to all output filenames    [default: ""]
  --exps       STR    Comma-separated sub-experiments to run  [default: D1,D2,D3,D4]
"""

import argparse
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CLI ────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Exp D: ES multi-step theory validation")
    # landscape
    p.add_argument("--d",        type=int,   default=500,    help="Parameter space dimension")
    p.add_argument("--k",        type=int,   default=20,     help="Number of active eigenmodes (curved)")
    p.add_argument("--lam_max",  type=float, default=5.0,    help="Largest eigenvalue")
    p.add_argument("--lam_min",  type=float, default=0.5,    help="Smallest (non-flat) eigenvalue")
    p.add_argument("--flat",     type=float, default=1e-4,   help="Flat eigenvalue for inactive dims")
    p.add_argument("--spectrum", type=str,   default="powerlaw",
                   choices=["uniform", "powerlaw", "range"],
                   help="Eigenvalue spectrum type")
    p.add_argument("--beta",     type=float, default=1.0,    help="Power-law exponent (spectrum=powerlaw)")
    # ES
    p.add_argument("--N",        type=int,   default=50,     help="ES population size")
    p.add_argument("--sigma",    type=float, default=0.1,    help="ES exploration std")
    p.add_argument("--xi",       type=float, default=0.0,    help="Observation noise std")
    p.add_argument("--ddof",     type=int,   default=0,      help="Std denominator (0=N, 1=N-1)")
    # simulation
    p.add_argument("--T",        type=int,   default=400,    help="Number of ES steps")
    p.add_argument("--n_trials", type=int,   default=200,    help="Number of independent trajectories")
    p.add_argument("--theta0_norm", type=float, default=10.0, help="‖θ_0‖ (starting distance from θ*)")
    p.add_argument("--seed",     type=int,   default=42,     help="Random seed")
    # GD sweep (D4)
    p.add_argument("--gd_lr",    type=float, default=None,   help="GD learning rate (default: 2/(lam_min+lam_max))")
    # output
    p.add_argument("--out_dir",  type=str,   default=None,   help="Output directory (overrides default)")
    p.add_argument("--tag",      type=str,   default="",     help="Optional tag appended to filenames")
    p.add_argument("--exps",     type=str,   default="D1,D2,D3,D4",
                   help="Comma-separated list of sub-experiments to run")
    return p.parse_args()


# ── paths ──────────────────────────────────────────────────────────────────────

def setup_paths(args):
    if args.out_dir:
        base = Path(args.out_dir)
    else:
        # prefer cluster path if it exists, else local
        cluster = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation")
        local   = Path.home() / "DL_Projects" / "EvolStrategyTheory_validation"
        base    = cluster if cluster.exists() else local
    figs = base / "figures"
    data = base / "data"
    figs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return figs, data


# ── landscape ─────────────────────────────────────────────────────────────────

def make_rotation(d, seed, device):
    """Random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    A = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    return (Q * signs.unsqueeze(0)).to(device)


def make_eigenvalues(d, k, lam_max, lam_min, flat, spectrum, beta):
    """Return 1-D tensor of eigenvalues (descending for active, flat for rest)."""
    eigs = torch.full((d,), flat, dtype=torch.float64)
    if k == 0:
        return eigs
    if spectrum == "uniform":
        eigs[:k] = lam_max
    elif spectrum == "powerlaw":
        idx = torch.arange(1, k + 1, dtype=torch.float64)
        vals = lam_max * (idx ** -beta)
        vals = vals.clamp(min=lam_min)
        eigs[:k] = vals
    elif spectrum == "range":
        eigs[:k] = torch.logspace(
            np.log10(lam_max), np.log10(lam_min), k, dtype=torch.float64
        )
    return eigs  # shape (d,)


def build_landscape(args, device):
    """Return rotation U (d×d) and eigenvalues lam (d,) on device."""
    U   = make_rotation(args.d, args.seed, device)
    lam = make_eigenvalues(args.d, args.k, args.lam_max, args.lam_min,
                           args.flat, args.spectrum, args.beta).to(device)
    return U, lam   # Q = U diag(lam) U^T


def Q_vec(U, lam, v):
    """Compute Q @ v efficiently: Q = U diag(lam) U^T, v shape (..., d)."""
    return (U @ (lam.unsqueeze(-1) * (U.T @ v.unsqueeze(-1)))).squeeze(-1)


def Q_mat_vec_batch(U, lam, V):
    """Q @ V  where V is (d, n) — returns (d, n)."""
    return U @ (lam.unsqueeze(-1) * (U.T @ V))


# ── ZScoreES step ─────────────────────────────────────────────────────────────

def zscore_es_step(theta, U, lam, sigma, alpha, N, xi, ddof, device):
    """
    One ZScoreES step from theta (shape: (d,)).
    Returns new_theta (d,) and diagnostics dict.
    """
    d = theta.shape[0]
    # sample perturbations
    eps = torch.randn(N, d, device=device, dtype=torch.float64)      # (N, d)
    # rewards: R_i = -½(θ+σε_i)^T Q (θ+σε_i)
    # = -½θ^T Q θ - σ v^T ε_i - ½σ² ε_i^T Q ε_i
    v = Q_vec(U, lam, theta)                                           # (d,)
    linear_term = -sigma * (eps @ v)                                   # (N,)
    Qeps = (U @ (lam.unsqueeze(-1) * (U.T @ eps.T))).T                # (N, d)
    quad_term = -0.5 * sigma**2 * (eps * Qeps).sum(-1)                # (N,)
    obs_noise = xi * torch.randn(N, device=device, dtype=torch.float64) if xi > 0 else 0.0
    R = linear_term + quad_term + obs_noise                            # (N,)
    # z-score
    mu_R = R.mean()
    if ddof == 0:
        sigma_R = R.std(correction=0).clamp(min=1e-12)
    else:
        sigma_R = R.std(correction=1).clamp(min=1e-12)
    Z = (R - mu_R) / sigma_R                                           # (N,)
    # update
    delta = alpha / N * (Z.unsqueeze(-1) * eps).sum(0)                # (d,)
    return theta + delta, {"sigma_R": sigma_R.item(), "v_norm": v.norm().item(), "delta": delta}


# ── GD step ───────────────────────────────────────────────────────────────────

def gd_step(theta, U, lam, lr, xi, device):
    """One gradient descent step (with optional obs noise on gradient)."""
    grad = Q_vec(U, lam, theta)   # ∇(-R) = Q θ
    if xi > 0:
        grad = grad + xi * torch.randn_like(grad)
    return theta - lr * grad


# ── theory predictions ────────────────────────────────────────────────────────

def theory_gamma(theta0, U, lam, sigma, alpha, xi):
    """Frozen-σ_R effective step size γ = ασ/σ_R(θ_0)."""
    v = Q_vec(U, lam, theta0)
    v_norm_sq = (v**2).sum().item()
    tr_Q2 = (lam**4).sum().item()  # Tr[Q²] = Σ λ_i²  ... wait, Tr[Q^2]=Σλ_i²
    sigma_R = np.sqrt(sigma**2 * v_norm_sq + 0.5 * sigma**4 * tr_Q2 + xi**2)
    gamma = alpha * sigma / max(sigma_R, 1e-12)
    return gamma, sigma_R


def theory_mean_trajectory(theta0, U, lam, gamma, T):
    """
    E[θ_t] = (I - γQ)^t θ_0  (frozen-γ approximation).
    In eigenbasis: E[c_i(t)] = (1 - γλ_i)^t  c_i(0)
    where c_i = u_i^T θ.
    Returns array of shape (T+1, d) in eigenbasis coordinates.
    """
    c0 = (U.T @ theta0).cpu().numpy()          # (d,) coords in eigenbasis
    lam_np = lam.cpu().numpy()
    decay  = (1.0 - gamma * lam_np)           # (d,)
    t_idx  = np.arange(T + 1)
    # c_i(t) = decay_i^t * c_i(0)
    coords = c0[None, :] * (decay[None, :] ** t_idx[:, None])  # (T+1, d)
    return coords, decay


def _sigma_R_at_zero(lam, sigma, xi):
    """σ_R when θ→0: only curvature and obs noise contribute."""
    tr_Q2 = (lam ** 4).sum().item()   # Tr[Q²] = Σ λ_i²
    return float(np.sqrt(0.5 * sigma**4 * tr_Q2 + xi**2 + 1e-30))


def theory_noise_floor_model0(lam, sigma, alpha, N, xi):
    """
    Model 0 — Simplified proposition (theory.tex new proposition):
      η_t ~ N(0, α²/N · I_d)   [isotropic only]
      Stationary Var[c_i(∞)] = (α²/N) / (1 - γ_k²)
      γ_k = 1 - (ασ/σ_R*)λ_k,  σ_R* = σ_R at θ=0
    """
    lam_np   = lam.cpu().numpy()
    sigma_R0 = _sigma_R_at_zero(lam, sigma, xi)
    gamma0   = alpha * sigma / max(sigma_R0, 1e-12)

    noise_var_i = np.full(len(lam_np), alpha**2 / N)             # α²/N everywhere
    denom       = np.maximum(1.0 - (1.0 - gamma0 * lam_np)**2, 1e-10)
    stat_vars   = noise_var_i / denom
    return stat_vars, gamma0, sigma_R0


def theory_noise_floor_model1(lam, sigma, alpha, N, xi):
    """
    Model 1 — Full Prop 3 Cov[Δθ], evaluated near θ*=0 (v→0):
      Cov[η_t] ≈ (α²/N)(I_d + 2σ⁴Q²/σ_R²)
      Per-mode noise: α²/N + 2σ⁴α²λ_i²/(N σ_R²)
    This is the anisotropic correction from the quadratic curvature term.
    """
    lam_np   = lam.cpu().numpy()
    sigma_R0 = _sigma_R_at_zero(lam, sigma, xi)
    gamma0   = alpha * sigma / max(sigma_R0, 1e-12)

    noise_iso   = alpha**2 / N
    noise_Q2_i  = 2.0 * sigma**4 * alpha**2 / (N * sigma_R0**2) * lam_np**2
    noise_var_i = noise_iso + noise_Q2_i
    denom       = np.maximum(1.0 - (1.0 - gamma0 * lam_np)**2, 1e-10)
    stat_vars   = noise_var_i / denom
    return stat_vars, gamma0, sigma_R0


def theory_noise_floor_model2_dynamic(traj_coords, sigma_R_traj, lam, alpha, N):
    """
    Model 2 — Dynamic γ: use the empirical σ_R(t) trajectory.
    Instead of a single frozen γ*, use the time-varying contraction.
    Computes the stationary variance by iterating the OU recursion
    using the observed σ_R values (averaged over trials).

    Var_i(t+1) = (1-γ_t λ_i)² Var_i(t) + (α²/N)
    where γ_t = ασ / mean_σ_R(t).

    NOTE: this is a *data-driven* model, not closed-form theory.
    """
    lam_np   = lam.cpu().numpy()
    T        = sigma_R_traj.shape[1]
    mean_sR  = sigma_R_traj.mean(axis=0)          # (T,)

    # Build variance trajectory per mode
    # Start from 0 (or could start from empirical variance at t=0)
    var_i = np.zeros(len(lam_np))
    alpha_val = alpha
    for t in range(T):
        gamma_t  = alpha_val * (2 * alpha_val) / max(mean_sR[t], 1e-12)
        # note: alpha = sigma/2, so ασ = α·2α = 2α²; γ_t = ασ/σ_R = 2α²/σ_R
        contract = (1.0 - gamma_t * lam_np) ** 2
        var_i    = contract * var_i + alpha_val**2 / N

    return var_i  # stationary variance per mode (after T steps)


# ── simulation: many trials ───────────────────────────────────────────────────

def run_es_trials(theta0, U, lam, args, device, n_trials=None):
    """
    Run `n_trials` independent ES trajectories for T steps.
    Returns:
      traj_coords: (n_trials, T+1, d)  — eigenbasis coordinates c_i(t) = u_i^T θ_t
      sigma_R_traj: (n_trials, T)      — σ_R at each step
    """
    if n_trials is None:
        n_trials = args.n_trials
    d  = args.d
    T  = args.T

    traj_coords = np.zeros((n_trials, T + 1, d), dtype=np.float32)
    sigma_R_traj = np.zeros((n_trials, T),        dtype=np.float32)

    # precompute U^T for projection
    UT = U.T  # (d, d)

    for trial in range(n_trials):
        theta = theta0.clone()
        c0 = (UT @ theta).cpu().numpy()
        traj_coords[trial, 0] = c0
        for t in range(T):
            theta, diag = zscore_es_step(theta, U, lam, args.sigma, args.sigma / 2.0,
                                          args.N, args.xi, args.ddof, device)
            traj_coords[trial, t + 1] = (UT @ theta).cpu().numpy()
            sigma_R_traj[trial, t]    = diag["sigma_R"]
        if (trial + 1) % 20 == 0:
            print(f"  ES trial {trial+1}/{n_trials}", flush=True)

    return traj_coords, sigma_R_traj


def run_gd_trajectory(theta0, U, lam, lr, T, xi, device):
    """Single GD trajectory for T steps."""
    d = theta0.shape[0]
    UT = U.T
    coords = np.zeros((T + 1, d), dtype=np.float64)
    coords[0] = (UT @ theta0).cpu().numpy()
    theta = theta0.clone().to(torch.float64)
    for t in range(T):
        theta = gd_step(theta, U, lam, lr, xi, device)
        coords[t + 1] = (UT @ theta).cpu().numpy()
    return coords


# ── plotting helpers ──────────────────────────────────────────────────────────

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

def savefig(fig, path, tag=""):
    stem = path.stem + (f"_{tag}" if tag else "")
    out  = path.parent / (stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}", flush=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# D1: Mean trajectory vs theory
# ══════════════════════════════════════════════════════════════════════════════

def exp_D1(traj_coords, sigma_R_traj, theta0, U, lam, args, gamma0, figs, data, tag):
    print("\n[D1] Mean trajectory vs frozen-σ_R theory", flush=True)
    T = args.T
    d = args.d
    t_ax = np.arange(T + 1)

    # theory mean trajectory in eigenbasis
    theory_coords, decay = theory_mean_trajectory(theta0, U, lam, gamma0, T)

    # empirical mean over trials
    emp_mean = traj_coords.mean(axis=0)   # (T+1, d)
    emp_std  = traj_coords.std(axis=0)    # (T+1, d)

    # ── 1a: ‖E[θ_t]‖ vs theory ‖(I-γQ)^t θ_0‖ ──
    emp_norm   = np.linalg.norm(emp_mean, axis=-1)   # (T+1,)
    theory_norm = np.linalg.norm(theory_coords, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(t_ax, emp_norm,    label="Empirical ‖E[θ_t]‖", lw=2)
    ax.plot(t_ax, theory_norm, label="Theory (frozen γ)",  lw=2, ls="--")
    ax.set_xlabel("Step t"); ax.set_ylabel("‖E[θ_t]‖")
    ax.set_title("D1a: Mean trajectory norm")
    ax.legend(); ax.grid(alpha=0.3)

    # ── 1b: per-mode decay for top-k modes ──
    ax = axes[1]
    n_show = min(6, args.k)
    for i in range(n_show):
        c = COLORS[i]
        ax.plot(t_ax, np.abs(emp_mean[:, i]),
                color=c, lw=1.5, label=f"λ={lam[i].item():.2f}")
        ax.plot(t_ax, np.abs(theory_coords[:, i]),
                color=c, lw=1.5, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Step t")
    ax.set_ylabel("|E[c_i(t)]|"); ax.set_title("D1b: Per-eigenmode mean (log)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(f"D1 Mean Trajectory  d={args.d} k={args.k} N={args.N} σ={args.sigma}")
    savefig(fig, figs / "expD1_mean_trajectory.png", tag)

    # ── save data ──
    res = {
        "emp_mean": emp_mean, "emp_std": emp_std,
        "theory_coords": theory_coords, "decay": decay,
        "gamma0": gamma0, "lam": lam.cpu().numpy(),
        "t_ax": t_ax, "args": vars(args),
    }
    pkl = data / f"expD1_mean_trajectory{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)

    # ── ratio theory/empirical — only where signal is above noise floor ──
    # Use early times where |E[c_i]| >> noise level
    noise_level = np.sqrt((args.sigma/2)**2 * args.d / args.N)   # rough noise scale
    for t_check in [T // 4, T // 2]:
        emp_mag = np.abs(emp_mean[t_check, :args.k])
        valid   = emp_mag > noise_level * 0.5
        ratio   = np.abs(theory_coords[t_check, :args.k]) / (emp_mag + 1e-12)
        print(f"  t={t_check}: theory/emp ratio (valid modes only): "
              f"{ratio[valid][:5].round(3)}  ({valid.sum()}/{args.k} modes above noise)", flush=True)

    return res


# ══════════════════════════════════════════════════════════════════════════════
# D2: Per-eigenmode decay rate validation
# ══════════════════════════════════════════════════════════════════════════════

def exp_D2(traj_coords, theta0, U, lam, args, gamma0, figs, data, tag):
    print("\n[D2] Per-eigenmode decay rates", flush=True)
    T = args.T
    t_ax = np.arange(T + 1)

    lam_np = lam.cpu().numpy()
    theory_decay_rates = np.log(np.abs(1.0 - gamma0 * lam_np))  # (d,)

    # empirical log |E[c_i(t)]|: fit slope
    emp_mean = traj_coords.mean(axis=0)   # (T+1, d)

    n_modes = min(args.k, 20)
    emp_rates = []
    fit_t = t_ax[1: T // 2]  # use first half for slope fit

    for i in range(n_modes):
        y = np.abs(emp_mean[fit_t, i]) + 1e-30
        log_y = np.log(y)
        # linear fit: log|c_i(t)| = rate_i * t + const
        A = np.vstack([fit_t, np.ones(len(fit_t))]).T
        rate, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
        emp_rates.append(rate)
    emp_rates = np.array(emp_rates)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # D2a: theory vs empirical decay rate
    ax = axes[0]
    mode_colors = plt.cm.tab20(np.linspace(0, 1, n_modes))
    ax.scatter(theory_decay_rates[:n_modes], emp_rates,
               c=mode_colors, s=60, zorder=5)
    lim_min = min(theory_decay_rates[:n_modes].min(), emp_rates.min()) * 1.1
    lim_max = max(theory_decay_rates[:n_modes].max(), emp_rates.max()) * 0.9
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="y=x")
    ax.set_xlabel("Theory log(1-γλ_i)"); ax.set_ylabel("Empirical slope")
    ax.set_title("D2a: Decay rate — theory vs empirical")
    ax.legend(); ax.grid(alpha=0.3)
    for i in range(n_modes):
        ax.annotate(f"λ={lam_np[i]:.2f}", (theory_decay_rates[i], emp_rates[i]),
                    fontsize=6, ha="left")

    # D2b: log |E[c_i(t)]| traces for top modes
    ax = axes[1]
    for i in range(min(6, n_modes)):
        c = COLORS[i]
        y = np.abs(emp_mean[:, i]) + 1e-30
        ax.plot(t_ax, np.log(y), color=c, lw=1.5, label=f"λ={lam_np[i]:.2f}")
        # theory line
        y_th = np.abs(emp_mean[0, i]) * np.exp(theory_decay_rates[i] * t_ax)
        ax.plot(t_ax, np.log(y_th + 1e-30), color=c, lw=1.5, ls="--")
    ax.set_xlabel("Step t"); ax.set_ylabel("log |E[c_i(t)]|")
    ax.set_title("D2b: Log decay traces (solid=emp, dash=theory)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(f"D2 Eigenmode Decay  d={args.d} k={args.k} N={args.N} σ={args.sigma} γ={gamma0:.4f}")
    savefig(fig, figs / "expD2_eigenmode_decay.png", tag)

    res = {
        "emp_rates": emp_rates, "theory_decay_rates": theory_decay_rates[:n_modes],
        "gamma0": gamma0, "lam": lam_np[:n_modes],
        "emp_mean": emp_mean, "t_ax": t_ax, "args": vars(args),
    }
    pkl = data / f"expD2_eigenmode_decay{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)

    print(f"  Theory rates (top-5): {theory_decay_rates[:5].round(4)}", flush=True)
    print(f"  Empirical rates (top-5): {emp_rates[:5].round(4)}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# D3: Noise floor — variance / stationary distribution
# ══════════════════════════════════════════════════════════════════════════════

def exp_D3(traj_coords, sigma_R_traj, theta0, U, lam, args, gamma0, figs, data, tag):
    """
    D3: Noise floor — compare three theory models against empirical Var[c_i(∞)].

    Model 0: simplified proposition  η ~ N(0, α²/N · I)
    Model 1: full Prop 3 Cov[Δθ]    isotropic + anisotropic Q² correction
    Model 2: dynamic γ               data-driven, uses empirical σ_R(t) trajectory
    """
    print("\n[D3] Noise floor — three-model theory comparison", flush=True)
    T   = args.T
    d   = args.d
    alpha = args.sigma / 2.0
    lam_np = lam.cpu().numpy()
    t_ax   = np.arange(T + 1)

    # ── Empirical quantities ──
    emp_var       = traj_coords.var(axis=0)              # (T+1, d)
    emp_var_total = emp_var.sum(axis=-1)                 # (T+1,)
    emp_norm2     = (traj_coords**2).sum(axis=-1).mean(axis=0)  # (T+1,)
    stationary_var_emp = emp_var[int(0.8 * T):].mean(axis=0)    # (d,) last 20%

    # ── Theory models ──
    sv_m0, gamma_star, sigma_R_star = theory_noise_floor_model0(
        lam, args.sigma, alpha, args.N, args.xi)
    sv_m1, _, _ = theory_noise_floor_model1(
        lam, args.sigma, alpha, args.N, args.xi)
    sv_m2 = theory_noise_floor_model2_dynamic(
        traj_coords, sigma_R_traj, lam, alpha, args.N)

    print(f"  γ* (frozen, near θ*): {gamma_star:.5f},  σ_R*: {sigma_R_star:.4f}", flush=True)
    print(f"  Empirical  ‖θ_T‖² (last 20%): {emp_norm2[int(0.8*T):].mean():.3f}", flush=True)
    print(f"  Model 0 (isotropic)    Σ Var: {sv_m0.sum():.3f}", flush=True)
    print(f"  Model 1 (+Q² correct.) Σ Var: {sv_m1.sum():.3f}", flush=True)
    print(f"  Model 2 (dynamic γ)    Σ Var: {sv_m2.sum():.3f}", flush=True)
    print(f"  Empirical  Σ Var[c_i]: {stationary_var_emp.sum():.3f}", flush=True)

    # ratios for top active modes
    n_act = min(args.k, 15)
    print(f"\n  Per-mode ratio (theory/empirical), top {n_act} active modes:", flush=True)
    print(f"  {'mode':>4} {'λ_i':>7} {'emp':>10} {'M0':>10} {'M1':>10} {'M2':>10} "
          f"{'M0/emp':>8} {'M1/emp':>8} {'M2/emp':>8}", flush=True)
    for i in range(n_act):
        ev = stationary_var_emp[i] + 1e-15
        print(f"  {i:>4} {lam_np[i]:>7.3f} {stationary_var_emp[i]:>10.4e} "
              f"{sv_m0[i]:>10.4e} {sv_m1[i]:>10.4e} {sv_m2[i]:>10.4e} "
              f"{sv_m0[i]/ev:>8.3f} {sv_m1[i]/ev:>8.3f} {sv_m2[i]/ev:>8.3f}",
              flush=True)

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # D3a: ‖θ_t‖² over time + theory floors
    ax = axes[0]
    ax.plot(t_ax, emp_norm2, lw=2, color="k", label="Empirical E[‖θ_t‖²]")
    ax.axhline(sv_m0.sum(), color="C0", ls="--", lw=1.5, label=f"M0 iso  Σ={sv_m0.sum():.2f}")
    ax.axhline(sv_m1.sum(), color="C1", ls="--", lw=1.5, label=f"M1 +Q²  Σ={sv_m1.sum():.2f}")
    ax.axhline(sv_m2.sum(), color="C2", ls=":",  lw=1.5, label=f"M2 dyn  Σ={sv_m2.sum():.2f}")
    ax.set_xlabel("Step t"); ax.set_ylabel("E[‖θ_t‖²]")
    ax.set_title("D3a: Convergence + three theory floors")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # D3b: Per-mode stationary variance — all models vs empirical
    ax = axes[1]
    mode_idx = np.arange(min(args.k + 5, d))
    ax.semilogy(mode_idx, stationary_var_emp[mode_idx], "ko-", ms=5, lw=2,   label="Empirical")
    ax.semilogy(mode_idx, sv_m0[mode_idx],              "C0x--", ms=5, lw=1.5, label="M0: iso α²/N")
    ax.semilogy(mode_idx, sv_m1[mode_idx],              "C1s--", ms=5, lw=1.5, label="M1: +Q² correction")
    ax.semilogy(mode_idx, sv_m2[mode_idx],              "C2^:",  ms=5, lw=1.5, label="M2: dynamic γ")
    ax.set_xlabel("Eigenmode index i"); ax.set_ylabel("Stationary Var[c_i(∞)]")
    ax.set_title("D3b: Per-mode stationary variance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # D3c: Ratio model/empirical per mode — how well does each theory fit?
    ax = axes[2]
    eps = 1e-15
    ratio_m0 = sv_m0[mode_idx] / (stationary_var_emp[mode_idx] + eps)
    ratio_m1 = sv_m1[mode_idx] / (stationary_var_emp[mode_idx] + eps)
    ratio_m2 = sv_m2[mode_idx] / (stationary_var_emp[mode_idx] + eps)
    ax.plot(mode_idx, ratio_m0, "C0o-", ms=5, lw=1.5, label="M0 / empirical")
    ax.plot(mode_idx, ratio_m1, "C1s-", ms=5, lw=1.5, label="M1 / empirical")
    ax.plot(mode_idx, ratio_m2, "C2^-", ms=5, lw=1.5, label="M2 / empirical")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="perfect = 1")
    ax.set_xlabel("Eigenmode index i"); ax.set_ylabel("Theory / Empirical ratio")
    ax.set_title("D3c: Theory fit ratio (1.0 = perfect)")
    ax.set_ylim(0, max(ratio_m0.max(), ratio_m1.max(), ratio_m2.max(), 2.0) * 1.1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"D3 Noise Floor Comparison  d={args.d} N={args.N} σ={args.sigma} ξ={args.xi}")
    savefig(fig, figs / "expD3_noise_floor_comparison.png", tag)

    res = {
        "emp_norm2": emp_norm2, "emp_var": emp_var,
        "stationary_var_emp": stationary_var_emp,
        "sv_model0": sv_m0, "sv_model1": sv_m1, "sv_model2": sv_m2,
        "gamma_star": gamma_star, "sigma_R_star": sigma_R_star,
        "ratio_m0": ratio_m0, "ratio_m1": ratio_m1, "ratio_m2": ratio_m2,
        "lam": lam_np, "t_ax": t_ax, "args": vars(args),
    }
    pkl = data / f"expD3_noise_floor_comparison{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# D4: ES vs GD comparison
# ══════════════════════════════════════════════════════════════════════════════

def exp_D4(traj_coords, theta0, U, lam, args, gamma0, device, figs, data, tag):
    print("\n[D4] ES vs GD comparison", flush=True)
    T    = args.T
    d    = args.d
    lam_np = lam.cpu().numpy()
    t_ax   = np.arange(T + 1)

    # ES: mean ‖θ‖ over trials
    emp_norm = np.sqrt((traj_coords**2).sum(axis=-1).mean(axis=0))   # (T+1,)

    # GD trajectories
    # optimal step: η* = 2 / (λ_max + λ_min)
    active_lam = lam_np[:args.k]
    lam_min_active = active_lam[active_lam > args.flat * 10].min() if (active_lam > args.flat * 10).any() else args.flat
    lam_max_active = active_lam.max()
    lr_opt   = args.gd_lr if args.gd_lr else 2.0 / (lam_min_active + lam_max_active)
    lr_half  = lr_opt / 2.0
    lr_2x    = min(lr_opt * 2.0, 1.9 / lam_max_active)  # cap at stability

    print(f"  GD lr_opt={lr_opt:.4f}  (λ_min={lam_min_active:.3f}, λ_max={lam_max_active:.3f})", flush=True)

    gd_coords_opt  = run_gd_trajectory(theta0, U, lam, lr_opt,  T, xi=0.0,     device=device)
    gd_coords_half = run_gd_trajectory(theta0, U, lam, lr_half, T, xi=0.0,     device=device)
    gd_coords_noisy= run_gd_trajectory(theta0, U, lam, lr_opt,  T, xi=args.xi, device=device)

    gd_norm_opt   = np.sqrt((gd_coords_opt  **2).sum(-1))
    gd_norm_half  = np.sqrt((gd_coords_half **2).sum(-1))
    gd_norm_noisy = np.sqrt((gd_coords_noisy**2).sum(-1))

    # ── steps comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.semilogy(t_ax, emp_norm,      lw=2,   label=f"ES N={args.N}")
    ax.semilogy(t_ax, gd_norm_opt,   lw=2,   label=f"GD η*={lr_opt:.3f}")
    ax.semilogy(t_ax, gd_norm_half,  lw=1.5, ls="--", label=f"GD η/2={lr_half:.3f}")
    ax.semilogy(t_ax, gd_norm_noisy, lw=1.5, ls=":",  label=f"GD+noise ξ={args.xi}")
    ax.set_xlabel("Steps t"); ax.set_ylabel("‖E[θ_t]‖")
    ax.set_title("D4a: Convergence vs steps")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── function evaluations (ES costs N per step, GD costs 1) ──
    ax = axes[1]
    fe_es  = np.arange(T + 1) * args.N
    fe_gd  = np.arange(T + 1)
    ax.semilogy(fe_es, emp_norm,     lw=2,   label=f"ES N={args.N}")
    ax.semilogy(fe_gd, gd_norm_opt,  lw=2,   label="GD (exact grad)")
    ax.semilogy(fe_gd, gd_norm_noisy,lw=1.5, ls="--", label="GD+obs noise")
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("‖E[θ_t]‖")
    ax.set_title("D4b: Convergence vs #evals")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── per-eigenmode final ‖θ_T‖² for ES vs GD ──
    ax = axes[2]
    n_show = min(args.k + 3, 30)
    es_final_var = traj_coords[:, -1, :n_show]**2           # (trials, n_show)
    gd_final     = gd_coords_opt[-1, :n_show]**2

    ax.semilogy(np.arange(n_show), es_final_var.mean(0), "o-", ms=4, lw=1.5,
                label="ES E[c_i²(T)]")
    ax.semilogy(np.arange(n_show), gd_final,             "s--", ms=4, lw=1.5,
                label="GD c_i²(T)")
    ax.set_xlabel("Eigenmode index i"); ax.set_ylabel("c_i²")
    ax.set_title("D4c: Residual per mode at T")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(np.arange(min(args.k, n_show)), lam_np[:min(args.k, n_show)],
             "^:", color="gray", ms=4, lw=1, alpha=0.6)
    ax2.set_ylabel("λ_i", color="gray"); ax2.tick_params(axis="y", colors="gray")

    fig.suptitle(f"D4 ES vs GD  d={args.d} k={args.k} N={args.N} σ={args.sigma}")
    savefig(fig, figs / "expD4_es_vs_gd.png", tag)

    # ── bonus: speedup factor at various ‖θ‖ thresholds ──
    thresholds = [0.5, 0.2, 0.1]
    theta0_norm = float(np.sqrt((traj_coords[:, 0, :]**2).sum(-1).mean()))
    print(f"  ‖θ_0‖ = {theta0_norm:.2f}", flush=True)
    for frac in thresholds:
        target = frac * theta0_norm
        def steps_to(arr):
            idx = np.where(arr < target)[0]
            return int(idx[0]) if len(idx) else T
        es_steps  = steps_to(emp_norm)
        gd_steps  = steps_to(gd_norm_opt)
        print(f"  Reach {frac:.0%} of ‖θ_0‖: ES={es_steps} steps ({es_steps*args.N} evals) "
              f"| GD={gd_steps} steps ({gd_steps} evals)", flush=True)

    res = {
        "emp_norm": emp_norm, "gd_norm_opt": gd_norm_opt,
        "gd_norm_half": gd_norm_half, "gd_norm_noisy": gd_norm_noisy,
        "gd_coords_opt": gd_coords_opt,
        "lr_opt": lr_opt, "lam_min_active": lam_min_active, "lam_max_active": lam_max_active,
        "fe_es": fe_es, "fe_gd": fe_gd,
        "t_ax": t_ax, "args": vars(args),
    }
    pkl = data / f"expD4_es_vs_gd{('_'+tag) if tag else ''}.pkl"
    with open(pkl, "wb") as f: pickle.dump(res, f)
    print(f"  Data: {pkl}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Config: d={args.d} k={args.k} N={args.N} σ={args.sigma} ξ={args.xi} "
          f"T={args.T} n_trials={args.n_trials} spectrum={args.spectrum}", flush=True)

    figs, data = setup_paths(args)
    tag = args.tag

    exps_to_run = [e.strip() for e in args.exps.split(",")]

    # ── Build landscape ──
    U, lam = build_landscape(args, device)
    lam = lam.to(torch.float64)
    U   = U.to(torch.float64)

    # ── Initial point θ_0 ──
    torch.manual_seed(args.seed + 1)
    theta0_raw = torch.randn(args.d, dtype=torch.float64, device=device)
    theta0 = theta0_raw / theta0_raw.norm() * args.theta0_norm

    # ── Theory quantities at θ_0 ──
    gamma0, sigma_R0 = theory_gamma(theta0, U, lam, args.sigma, args.sigma / 2.0, args.xi)
    print(f"Theory γ₀ = {gamma0:.5f},  σ_R(θ₀) = {sigma_R0:.4f}", flush=True)
    print(f"‖θ_0‖ = {theta0.norm().item():.2f}", flush=True)

    # ── Run ES simulation (shared across D1-D4) ──
    print(f"\nRunning {args.n_trials} ES trials × {args.T} steps...", flush=True)
    t0 = time.time()
    traj_coords, sigma_R_traj = run_es_trials(theta0, U, lam, args, device)
    print(f"Simulation done in {time.time()-t0:.1f}s", flush=True)

    # Save raw simulation data
    sim_pkl = data / f"expD_simulation{('_'+tag) if tag else ''}.pkl"
    with open(sim_pkl, "wb") as f:
        pickle.dump({
            "traj_coords": traj_coords,
            "sigma_R_traj": sigma_R_traj,
            "theta0": theta0.cpu().numpy(),
            "lam": lam.cpu().numpy(),
            "gamma0": gamma0,
            "sigma_R0": sigma_R0,
            "args": vars(args),
        }, f)
    print(f"Simulation saved: {sim_pkl}", flush=True)

    # ── Sub-experiments ──
    if "D1" in exps_to_run:
        exp_D1(traj_coords, sigma_R_traj, theta0, U, lam, args, gamma0, figs, data, tag)

    if "D2" in exps_to_run:
        exp_D2(traj_coords, theta0, U, lam, args, gamma0, figs, data, tag)

    if "D3" in exps_to_run:
        exp_D3(traj_coords, sigma_R_traj, theta0, U, lam, args, gamma0, figs, data, tag)

    if "D4" in exps_to_run:
        exp_D4(traj_coords, theta0, U, lam, args, gamma0, device, figs, data, tag)

    print(f"\nAll done. Figures → {figs}, Data → {data}", flush=True)


if __name__ == "__main__":
    main()
