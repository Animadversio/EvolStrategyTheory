"""
Comparison of full ES simulation vs self-consistent (mean-field) equations
on a diagonal quadratic landscape.

ES update rule:
    theta_{t+1} = theta_t + alpha * (1/N) * sum_i Z_i * epsilon_i
    epsilon_i ~ N(0, I_d),  Z_i = (R_i - mu_R) / sigma_R
    R_i = R(theta_t + sigma * epsilon_i),  alpha = sigma / 2

Self-consistent mean-field equations (moment closure):
    m_{k,t+1}    = (1 - alpha*sigma/sigma_R_bar * lambda_k) * m_{k,t}
    v_{k,t+1}    = (1 - alpha*sigma/sigma_R_bar * lambda_k)^2 * v_{k,t} + alpha^2/N
    sigma_R_bar  = sqrt(sigma^2 * sum_k lambda_k^2*(m_k^2+v_k)
                        + sigma^4/2 * sum_k lambda_k^2 + sigma_xi^2)
where m_{k,t} = E[x_{k,t}], v_{k,t} = Var[x_{k,t}], x_{k,t} = u_k^T theta_t.

Usage examples:
    # default (50-dim, step spectrum)
    python simulate_selfconsistent.py

    # 1000-dim power-law spectrum
    python simulate_selfconsistent.py --dim 1000 --spectrum powerlaw --exponent 1.0 \\
        --sigma 0.3 --N 30 --T 500 --n-traj 100

    # 1000-dim 1/k spectrum (exponent=1 alias)
    python simulate_selfconsistent.py --dim 1000 --spectrum powerlaw --exponent 1 \\
        --n-modes-plot 8 --out figures/selfconsistent_dynamics/powerlaw1000.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Vector fonts (Type 42 / TrueType embedded) + clean spine style
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Spectrum builders ─────────────────────────────────────────────────────────

def make_spectrum(dim: int, kind: str, **kw) -> np.ndarray:
    """
    Build an eigenvalue spectrum of length `dim`.

    kind='step'      : n_rel large values then n_flat small values
                       kw: n_rel, large_vals (list), flat_val
    kind='powerlaw'  : lambda_k = scale / k^exponent  for k=1..dim
                       kw: exponent (default 1.0), scale (default 1.0)
    kind='geometric' : lambda_k = scale * ratio^(k-1)
                       kw: ratio (default 0.5), scale (default 1.0)
    """
    k = np.arange(1, dim + 1, dtype=float)
    if kind == 'step':
        n_rel     = kw.get('n_rel', 5)
        large_val = kw.get('large_vals', np.array([10., 5., 2., 1., 0.5]))
        flat_val  = kw.get('flat_val', 0.05)
        large_val = np.asarray(large_val)[:n_rel]
        if len(large_val) < n_rel:
            large_val = np.pad(large_val, (0, n_rel - len(large_val)),
                               constant_values=large_val[-1])
        return np.concatenate([large_val, np.full(dim - n_rel, flat_val)])
    elif kind == 'powerlaw':
        exponent = kw.get('exponent', 1.0)
        scale    = kw.get('scale', 1.0)
        return scale / k ** exponent
    elif kind == 'geometric':
        ratio = kw.get('ratio', 0.5)
        scale = kw.get('scale', 1.0)
        return scale * ratio ** (k - 1)
    else:
        raise ValueError(f"Unknown spectrum kind: {kind!r}")


# ── Landscape ─────────────────────────────────────────────────────────────────

def reward(theta: np.ndarray, eigvals: np.ndarray) -> float:
    """Quadratic reward R(theta) = -1/2 * theta^T diag(eigvals) theta."""
    return -0.5 * float(np.dot(eigvals, theta ** 2))


# ── Full ES simulation ────────────────────────────────────────────────────────

def es_step(theta: np.ndarray, eigvals: np.ndarray,
            sigma: float, N: int,
            sigma_xi: float = 0.0,
            rng: np.random.Generator = None) -> np.ndarray:
    """One ES update step. Returns new theta."""
    if rng is None:
        rng = np.random.default_rng()

    alpha = 0.5 * sigma
    eps   = rng.standard_normal((N, len(theta)))              # (N, d)
    R     = -0.5 * ((theta + sigma * eps) ** 2 @ eigvals)     # vectorised
    if sigma_xi > 0:
        R += sigma_xi * rng.standard_normal(N)

    std_R = R.std()
    if std_R < 1e-12:
        return theta.copy()

    Z = (R - R.mean()) / std_R                                # (N,)
    return theta + alpha / N * (Z @ eps)                      # (d,)


def run_es(eigvals: np.ndarray, theta0: np.ndarray,
           sigma: float, N: int, T: int,
           n_traj: int = 200, sigma_xi: float = 0.0,
           seed: int = 0) -> np.ndarray:
    """
    Run n_traj independent ES trajectories for T steps.
    Returns theta_all of shape (T+1, n_traj, d).
    """
    rng = np.random.default_rng(seed)
    d   = len(eigvals)
    theta_all        = np.empty((T + 1, n_traj, d))
    theta_all[0]     = theta0

    for j in range(n_traj):
        th = theta0.copy()
        for t in range(T):
            th = es_step(th, eigvals, sigma, N,
                         sigma_xi=sigma_xi, rng=rng)
            theta_all[t + 1, j] = th
        if (j + 1) % max(1, n_traj // 5) == 0:
            print(f"  trajectories: {j+1}/{n_traj}")

    return theta_all


# ── Self-consistent mean-field equations ─────────────────────────────────────

def mf_step(m: np.ndarray, v: np.ndarray,
            eigvals: np.ndarray, sigma: float, N: int,
            sigma_xi: float = 0.0):
    """One step of the mean-field moment equations. Returns (m_new, v_new, sigma_R_bar)."""
    alpha = 0.5 * sigma
    lam2  = eigvals ** 2

    sigma_R_sq  = (sigma ** 2 * np.dot(lam2, m ** 2 + v)
                   + 0.5 * sigma ** 4 * lam2.sum()
                   + sigma_xi ** 2)
    sigma_R_bar = np.sqrt(max(sigma_R_sq, 1e-30))

    c = 1.0 - alpha * sigma / sigma_R_bar * eigvals
    return c * m, c ** 2 * v + alpha ** 2 / N, sigma_R_bar


def run_mf(eigvals: np.ndarray, theta0: np.ndarray,
           sigma: float, N: int, T: int,
           sigma_xi: float = 0.0):
    """
    Integrate the mean-field equations for T steps.
    Returns m_traj (T+1, d), v_traj (T+1, d), sigmaR_traj (T+1,).
    """
    d    = len(eigvals)
    m    = theta0.astype(float).copy()
    v    = np.zeros(d)
    lam2 = eigvals ** 2

    m_traj  = np.empty((T + 1, d))
    v_traj  = np.empty((T + 1, d))
    sR_traj = np.empty(T + 1)

    m_traj[0]  = m
    v_traj[0]  = v
    sR_traj[0] = np.sqrt(sigma ** 2 * np.dot(lam2, m ** 2)
                         + 0.5 * sigma ** 4 * lam2.sum() + sigma_xi ** 2)

    for t in range(T):
        m, v, sR      = mf_step(m, v, eigvals, sigma, N, sigma_xi=sigma_xi)
        m_traj[t + 1]  = m
        v_traj[t + 1]  = v
        sR_traj[t + 1] = sR

    return m_traj, v_traj, sR_traj


# ── Observable helpers ────────────────────────────────────────────────────────

def expected_reward_mf(m: np.ndarray, v: np.ndarray,
                       eigvals: np.ndarray) -> float:
    return -0.5 * float(np.dot(eigvals, m ** 2 + v))


def theoretical_sigmaR(theta: np.ndarray, eigvals: np.ndarray,
                       sigma: float, sigma_xi: float = 0.0) -> float:
    lam2 = eigvals ** 2
    return np.sqrt(sigma ** 2 * np.dot(lam2, theta ** 2)
                   + 0.5 * sigma ** 4 * lam2.sum()
                   + sigma_xi ** 2)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(eigvals, theta_all, m_traj, v_traj, sR_mf,
                 sigma, N, T, n_traj, n_modes_plot,
                 spectrum_label, out):
    d     = len(eigvals)
    steps = np.arange(T + 1)

    sim_mean = theta_all.mean(axis=1)   # (T+1, d)
    sim_var  = theta_all.var(axis=1)    # (T+1, d)

    ER_mf  = np.array([expected_reward_mf(m_traj[t], v_traj[t], eigvals)
                        for t in range(T + 1)])
    ER_sim = np.mean(
        -0.5 * (theta_all ** 2 @ eigvals), axis=1)   # (T+1,)

    # sigma_R: theoretical value averaged over trajectories
    lam2       = eigvals ** 2
    sR_sq_all  = (sigma ** 2 * (theta_all ** 2 @ lam2)
                  + 0.5 * sigma ** 4 * lam2.sum())    # (T+1, n_traj)
    sR_all     = np.sqrt(np.maximum(sR_sq_all, 0))
    sR_sim_mean = sR_all.mean(axis=1)
    sR_sim_std  = sR_all.std(axis=1)

    # Choose modes to display: log-spaced indices across the spectrum
    idx         = np.unique(np.round(
                      np.geomspace(1, d, n_modes_plot) - 1
                  ).astype(int))
    mode_labels = [rf"$k={k+1},\,\lambda={eigvals[k]:.2g}$" for k in idx]
    colors      = plt.cm.turbo(np.linspace(0.05, 0.95, len(idx)))

    fig = plt.figure(figsize=(15, 13))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    def _add_mode_panel(ax, ys_sim, ys_mf, ylabel, title, logy=False):
        for k, lbl, col in zip(idx, mode_labels, colors):
            fn = ax.semilogy if logy else ax.plot
            fn(steps, np.abs(ys_sim[:, k]) + 1e-15, color=col, lw=1.5, label=lbl)
            fn(steps, np.abs(ys_mf[:, k])  + 1e-15, color=col, lw=1.5, ls='--')
        ax.set_xlabel("step $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2, title="solid=sim  dashed=MF",
                  title_fontsize=6)

    _add_mode_panel(fig.add_subplot(gs[0, 0]),
                    sim_mean, m_traj,
                    r"$m_{k,t}$", "Mean per mode")

    _add_mode_panel(fig.add_subplot(gs[0, 1]),
                    sim_var, v_traj,
                    r"$v_{k,t}$", "Variance per mode", logy=True)

    _add_mode_panel(fig.add_subplot(gs[1, 0]),
                    sim_mean ** 2, m_traj ** 2,
                    r"$m_{k,t}^2$ (log)", r"$m_{k,t}^2$ — convergence rate",
                    logy=True)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, sR_sim_mean, 'b-', lw=2,
            label=r'sim $\langle\sigma_R(\theta_t)\rangle$')
    ax.fill_between(steps,
                    sR_sim_mean - sR_sim_std,
                    sR_sim_mean + sR_sim_std,
                    alpha=0.2, color='b', label=r'$\pm1\sigma$')
    ax.plot(steps, sR_mf, 'r--', lw=2,
            label=r'MF $\bar{\sigma}_{R,t}$')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\sigma_R$")
    ax.set_title(r"Reward std $\sigma_{R,t}$")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(steps, ER_sim, 'b-',  lw=2,
            label=r'sim $\langle R(\theta_t)\rangle$')
    ax.plot(steps, ER_mf,  'r--', lw=2,
            label=r'MF $\mathbb{E}[R]=-\frac{1}{2}\sum_k\lambda_k(m_k^2+v_k)$')
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\mathbb{E}[R(\theta_t)]$")
    ax.set_title("Expected reward: simulation vs mean-field")
    ax.legend(fontsize=9)

    fig.suptitle(
        rf"ES dynamics — {spectrum_label},  $d={d}$, $N={N}$, "
        rf"$\sigma={sigma}$, $n_{{\mathrm{{traj}}}}={n_traj}$"
        "\n"
        r"solid = MC simulation,  dashed = self-consistent mean-field",
        fontsize=11)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stem = os.path.splitext(out)[0]
    plt.savefig(stem + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(stem + ".pdf",           bbox_inches='tight')
    print(f"Saved {stem}.png  +  {stem}.pdf")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate ES dynamics and compare to self-consistent equations.")

    # Spectrum
    p.add_argument("--dim",      type=int,   default=50,
                   help="Dimensionality d (default: 50)")
    p.add_argument("--spectrum", type=str,   default="step",
                   choices=["step", "powerlaw", "geometric"],
                   help="Eigenspectrum type (default: step)")
    p.add_argument("--exponent", type=float, default=1.0,
                   help="Power-law exponent for --spectrum powerlaw (default: 1.0 = 1/k)")
    p.add_argument("--scale",    type=float, default=1.0,
                   help="Scale factor for spectrum (default: 1.0)")
    p.add_argument("--ratio",    type=float, default=0.5,
                   help="Decay ratio for --spectrum geometric (default: 0.5)")
    # Step-spectrum specific
    p.add_argument("--n-rel",    type=int,   default=5,
                   help="Number of relevant modes for step spectrum (default: 5)")
    p.add_argument("--flat-val", type=float, default=0.05,
                   help="Flat eigenvalue for step spectrum (default: 0.05)")

    # Initial condition
    p.add_argument("--init-scale", type=float, default=1.0,
                   help="Initial |theta_k| for the first n-rel modes (default: 1.0)")

    # ES hyperparameters
    p.add_argument("--sigma",    type=float, default=0.5,
                   help="Perturbation std sigma (default: 0.5)")
    p.add_argument("--N",        type=int,   default=30,
                   help="Population size N (default: 30)")
    p.add_argument("--T",        type=int,   default=300,
                   help="Number of ES steps (default: 300)")
    p.add_argument("--n-traj",   type=int,   default=200,
                   help="Number of MC trajectories (default: 200)")
    p.add_argument("--sigma-xi", type=float, default=0.0,
                   help="Observation noise std (default: 0.0)")
    p.add_argument("--seed",     type=int,   default=42)

    # Plot
    p.add_argument("--n-modes-plot", type=int, default=6,
                   help="Number of modes to show in panels (log-spaced, default: 6)")
    p.add_argument("--out", type=str,
                   default="figures/selfconsistent_dynamics/es_mf_comparison.png",
                   help="Output figure path")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    d    = args.dim

    # Build spectrum
    if args.spectrum == "step":
        eigvals = make_spectrum(d, "step",
                                n_rel=args.n_rel, flat_val=args.flat_val)
        spectrum_label = (rf"step spectrum ($n_{{rel}}={args.n_rel}$, "
                          rf"$\lambda_{{flat}}={args.flat_val}$)")
    elif args.spectrum == "powerlaw":
        eigvals = make_spectrum(d, "powerlaw",
                                exponent=args.exponent, scale=args.scale)
        spectrum_label = rf"power-law $\lambda_k = {args.scale}/k^{{{args.exponent}}}$"
    else:  # geometric
        eigvals = make_spectrum(d, "geometric",
                                ratio=args.ratio, scale=args.scale)
        spectrum_label = rf"geometric $\lambda_k = {args.scale}\cdot{args.ratio}^{{k-1}}$"

    # Initial condition: first n_rel modes displaced, rest zero
    theta0 = np.zeros(d)
    n_init = min(args.n_rel if args.spectrum == "step" else d, d)
    theta0[:n_init] = args.init_scale

    print(f"\nSpectrum : {spectrum_label}")
    print(f"d={d}  N={args.N}  sigma={args.sigma}  T={args.T}  "
          f"n_traj={args.n_traj}  seed={args.seed}")
    print(f"eigvals  : min={eigvals.min():.3g}  max={eigvals.max():.3g}  "
          f"sum={eigvals.sum():.3g}  Tr[Q^2]={np.dot(eigvals,eigvals):.3g}")

    print("\nRunning full ES simulation ...")
    theta_all = run_es(eigvals, theta0, args.sigma, args.N, args.T,
                       n_traj=args.n_traj, sigma_xi=args.sigma_xi,
                       seed=args.seed)

    print("Running mean-field equations ...")
    m_traj, v_traj, sR_mf = run_mf(eigvals, theta0, args.sigma, args.N, args.T,
                                    sigma_xi=args.sigma_xi)

    plot_results(eigvals, theta_all, m_traj, v_traj, sR_mf,
                 sigma=args.sigma, N=args.N, T=args.T, n_traj=args.n_traj,
                 n_modes_plot=args.n_modes_plot,
                 spectrum_label=spectrum_label,
                 out=args.out)


if __name__ == "__main__":
    main()
