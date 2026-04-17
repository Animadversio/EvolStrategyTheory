"""
Null-space experiment: how does a true null space (lambda_k = 0) affect ES dynamics?

Key theoretical prediction (from self-consistent equations):
  - Null modes have contraction factor 1 → pure random walk: v_k(t) = t * alpha^2/N
  - Null modes do NOT contribute to sigma_R (lambda_k^2 = 0)
  - Relevant modes are therefore FULLY DECOUPLED from the null space

This script verifies that prediction by sweeping null-space fraction
    [0%, 50%, 90%, 95%, 99%, 99.9%]
and showing that the relevant-mode dynamics are invariant to null fraction.

Usage:
    python simulate_nullspace.py                        # default sweep
    python simulate_nullspace.py --n-rel 5 --T 400 --n-traj 80
    python simulate_nullspace.py --null-fracs 0.0 0.9 0.99 --sigma 0.4
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── ES core (same as simulate_selfconsistent.py) ──────────────────────────────

def es_step(theta, eigvals, sigma, N, rng):
    alpha = 0.5 * sigma
    eps   = rng.standard_normal((N, len(theta)))
    R     = -0.5 * ((theta + sigma * eps) ** 2 @ eigvals)
    std_R = R.std()
    if std_R < 1e-12:
        return theta.copy()
    Z = (R - R.mean()) / std_R
    return theta + alpha / N * (Z @ eps)


def run_es(eigvals, theta0, sigma, N, T, n_traj, seed=0):
    rng = np.random.default_rng(seed)
    d   = len(eigvals)
    out = np.empty((T + 1, n_traj, d))
    out[0] = theta0
    for j in range(n_traj):
        th = theta0.copy()
        for t in range(T):
            th = es_step(th, eigvals, sigma, N, rng)
            out[t + 1, j] = th
    return out


# ── Mean-field equations ──────────────────────────────────────────────────────

def run_mf(eigvals, theta0, sigma, N, T):
    alpha = 0.5 * sigma
    lam2  = eigvals ** 2
    d     = len(eigvals)
    m     = theta0.astype(float).copy()
    v     = np.zeros(d)

    m_traj  = np.empty((T + 1, d))
    v_traj  = np.empty((T + 1, d))
    sR_traj = np.empty(T + 1)

    m_traj[0]  = m
    v_traj[0]  = v
    sR_traj[0] = np.sqrt(sigma ** 2 * np.dot(lam2, m ** 2)
                         + 0.5 * sigma ** 4 * lam2.sum())

    for t in range(T):
        sR_sq = (sigma ** 2 * np.dot(lam2, m ** 2 + v)
                 + 0.5 * sigma ** 4 * lam2.sum())
        sR    = np.sqrt(max(sR_sq, 1e-30))
        c     = 1.0 - alpha * sigma / sR * eigvals
        m     = c * m
        v     = c ** 2 * v + alpha ** 2 / N
        m_traj[t + 1]  = m
        v_traj[t + 1]  = v
        sR_traj[t + 1] = sR

    return m_traj, v_traj, sR_traj


# ── Experiment ────────────────────────────────────────────────────────────────

def build_config(null_frac, eigvals_rel, init_scale):
    """
    Build (eigvals, theta0) for a given null fraction.
    n_rel relevant modes with given eigvals, rest are exact zeros.
    """
    n_rel = len(eigvals_rel)
    # null_frac = n_null / d  →  d = n_rel / (1 - null_frac)
    d     = max(n_rel, round(n_rel / (1.0 - null_frac)))
    n_null = d - n_rel
    eigvals = np.concatenate([eigvals_rel, np.zeros(n_null)])
    theta0  = np.zeros(d)
    theta0[:n_rel] = init_scale
    return eigvals, theta0, d, n_null


def run_one(null_frac, eigvals_rel, init_scale, sigma, N, T, n_traj, seed):
    eigvals, theta0, d, n_null = build_config(null_frac, eigvals_rel, init_scale)
    print(f"  null_frac={null_frac:.1%}  d={d}  n_null={n_null}", flush=True)

    theta_all            = run_es(eigvals, theta0, sigma, N, T, n_traj, seed)
    m_traj, v_traj, sR_mf = run_mf(eigvals, theta0, sigma, N, T)

    # Relevant-mode statistics (first n_rel indices)
    n_rel = len(eigvals_rel)
    sim_mean_rel = theta_all.mean(axis=1)[:, :n_rel]   # (T+1, n_rel)
    sim_var_rel  = theta_all.var(axis=1)[:, :n_rel]    # (T+1, n_rel)

    # Null-mode variance (average over all null modes; index n_rel onward)
    if n_null > 0:
        sim_var_null = theta_all.var(axis=1)[:, n_rel:].mean(axis=1)  # (T+1,)
        mf_var_null  = v_traj[:, n_rel:].mean(axis=1)
    else:
        sim_var_null = np.zeros(T + 1)
        mf_var_null  = np.zeros(T + 1)

    # Reward
    ER_sim = np.mean(-0.5 * (theta_all ** 2 @ eigvals), axis=1)  # (T+1,)
    ER_mf  = -0.5 * (m_traj ** 2 + v_traj) @ eigvals              # (T+1,)

    # sigma_R
    lam2 = eigvals ** 2
    sR_sim = np.sqrt(np.maximum(
        sigma ** 2 * (theta_all ** 2 @ lam2).mean(axis=1)
        + 0.5 * sigma ** 4 * lam2.sum(), 0))

    return dict(
        sim_mean_rel=sim_mean_rel, sim_var_rel=sim_var_rel,
        mf_mean_rel=m_traj[:, :n_rel], mf_var_rel=v_traj[:, :n_rel],
        sim_var_null=sim_var_null, mf_var_null=mf_var_null,
        ER_sim=ER_sim, ER_mf=ER_mf,
        sR_sim=sR_sim, sR_mf=sR_mf,
        d=d, n_null=n_null,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sweep(results, null_fracs, eigvals_rel, sigma, N, T, n_traj, out):
    n_rel  = len(eigvals_rel)
    steps  = np.arange(T + 1)
    alpha  = 0.5 * sigma

    # Color per null fraction
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(null_fracs)))
    frac_labels = [f"{f:.1%}" for f in null_fracs]

    fig = plt.figure(figsize=(15, 13))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # ── (a) Mean of top relevant mode ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for res, frac, col, lbl in zip(results, null_fracs, colors, frac_labels):
        ax.plot(steps, res["sim_mean_rel"][:, 0],
                color=col, lw=1.5, label=lbl)
        ax.plot(steps, res["mf_mean_rel"][:, 0],
                color=col, lw=1.5, ls='--')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$m_{1,t}$ (top mode)")
    ax.set_title(rf"Mean of top relevant mode ($\lambda={eigvals_rel[0]:.4g}$)"
                 "\nsolid=sim  dashed=MF")
    ax.legend(fontsize=7, title="null fraction", title_fontsize=7)

    # ── (b) Variance of top relevant mode ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for res, frac, col, lbl in zip(results, null_fracs, colors, frac_labels):
        ax.semilogy(steps, res["sim_var_rel"][:, 0] + 1e-16,
                    color=col, lw=1.5, label=lbl)
        ax.semilogy(steps, res["mf_var_rel"][:, 0]  + 1e-16,
                    color=col, lw=1.5, ls='--')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$v_{1,t}$ (log)")
    ax.set_title("Variance of top relevant mode")
    ax.legend(fontsize=7, title="null fraction", title_fontsize=7)

    # ── (c) Null-mode variance (random walk) ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    rw_theory = (alpha ** 2 / N) * steps           # linear random-walk theory
    ax.plot(steps, rw_theory, 'k:', lw=2,
            label=r"theory: $\alpha^2/N \cdot t$")
    for res, frac, col, lbl in zip(results, null_fracs, colors, frac_labels):
        if res["n_null"] == 0:
            continue
        ax.plot(steps, res["sim_var_null"], color=col, lw=1.5, label=lbl)
        ax.plot(steps, res["mf_var_null"],  color=col, lw=1.5, ls='--')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\langle v_{k,t}\rangle_{\mathrm{null}}$")
    ax.set_title(r"Null-mode variance $\equiv$ free random walk")
    ax.legend(fontsize=7, title="null fraction", title_fontsize=7)

    # ── (d) sigma_R over time ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for res, frac, col, lbl in zip(results, null_fracs, colors, frac_labels):
        ax.plot(steps, res["sR_sim"], color=col, lw=1.5, label=lbl)
        ax.plot(steps, res["sR_mf"],  color=col, lw=1.5, ls='--')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\sigma_{R,t}$")
    ax.set_title(r"$\sigma_{R,t}$: unaffected by null space")
    ax.legend(fontsize=7, title="null fraction", title_fontsize=7)

    # ── (e) Expected reward ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    for res, frac, col, lbl in zip(results, null_fracs, colors, frac_labels):
        ax.plot(steps, res["ER_sim"], color=col, lw=1.5, label=lbl)
        ax.plot(steps, res["ER_mf"],  color=col, lw=1.5, ls='--')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\mathbb{E}[R(\theta_t)]$")
    ax.set_title("Expected reward: identical convergence across null fractions\n"
                 "solid=sim  dashed=MF")
    ax.legend(fontsize=7, title="null fraction", title_fontsize=7, ncol=3)

    fig.suptitle(
        rf"Null-space decoupling: $n_{{\mathrm{{rel}}}}={n_rel}$, "
        rf"$N={N}$, $\sigma={sigma}$, $n_{{\mathrm{{traj}}}}={n_traj}$"
        "\n"
        r"Relevant-mode dynamics are invariant to null-space fraction  "
        r"(solid=sim, dashed=MF)",
        fontsize=11)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stem = os.path.splitext(out)[0]
    plt.savefig(stem + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(stem + ".pdf",           bbox_inches='tight')
    print(f"Saved {stem}.png  +  {stem}.pdf")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep null-space fraction and verify decoupling from relevant modes.")
    p.add_argument("--n-rel",      type=int,   default=5,
                   help="Number of relevant modes (default: 5)")
    p.add_argument("--eigscale",   type=float, default=10.0,
                   help="Scale for power-law relevant eigenvalues (default: 10)")
    p.add_argument("--null-fracs", type=float, nargs="+",
                   default=[0.0, 0.5, 0.9, 0.95, 0.99, 0.999],
                   help="Null-space fractions to sweep")
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--sigma",      type=float, default=0.3)
    p.add_argument("--N",          type=int,   default=30)
    p.add_argument("--T",          type=int,   default=400)
    p.add_argument("--n-traj",     type=int,   default=100)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out", type=str,
                   default="figures/selfconsistent_dynamics/es_nullspace_sweep.png")
    return p.parse_args()


def main():
    args = parse_args()

    # Relevant eigenvalues: power law 1/k scaled
    k_rel      = np.arange(1, args.n_rel + 1, dtype=float)
    eigvals_rel = args.eigscale / k_rel

    print(f"Relevant eigenvalues: {eigvals_rel}")
    print(f"sigma={args.sigma}  N={args.N}  T={args.T}  n_traj={args.n_traj}")
    print(f"Null fractions: {args.null_fracs}\n")

    results = []
    for frac in args.null_fracs:
        res = run_one(frac, eigvals_rel, args.init_scale,
                      args.sigma, args.N, args.T, args.n_traj, args.seed)
        results.append(res)

    plot_sweep(results, args.null_fracs, eigvals_rel,
               args.sigma, args.N, args.T, args.n_traj, args.out)


if __name__ == "__main__":
    main()
