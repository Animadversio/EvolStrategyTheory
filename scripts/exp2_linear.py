#!/usr/bin/env python3
"""Exp 2: Linear landscape — on-manifold fraction (Prop 2).

Landscape: R(θ) = -v^T θ + ξ·noise,   v is fixed gradient direction
ZScoreES update: Δθ = α/N · Σ Z_i ε_i

Prop 2: On-manifold fraction
  ρ = E[||Δθ_∥||²] / E[||Δθ||²] = (1 + (N+1)·s) / (d + (N+1)·s)
where s = SNR = σ²||v||² / (σ²||v||² + ξ²)

Sub-experiments:
  2a: ρ vs reward noise ξ (SNR sweep), d ∈ {100, 1000}
  2b: ρ vs N, d=1000, ξ=0.1
  2c: ρ vs d, N=50, ξ=0.1
  2d: Mean cosine alignment of Δθ with gradient direction
  2e: Multi-step convergence on linear landscape
"""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import seaborn as sns
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType → editable text in PDF
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'


def save_fig(fig, stem):
    """Save figure as both PNG (150 dpi) and PDF (vector fonts)."""
    fig.savefig(FIGS / f'{stem}.png', dpi=150, bbox_inches='tight')
    fig.savefig(FIGS / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  → saved {stem}.png + .pdf", flush=True)
import pandas as pd

STORE = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2")
FIGS  = STORE / "figures"
DATA  = STORE / "data"
for p in [FIGS, DATA]: p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA  = 0.1
ALPHA  = SIGMA / 2.0
print(f"Device: {DEVICE}", flush=True)

T0_TOTAL = time.time()


def gc():
    if DEVICE.type == 'cuda':
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()


def safe_chunk(d, N, gb=1.5):
    return max(1, int(gb * 1e9 / (4 * N * d)))


# ── theory ────────────────────────────────────────────────────────────────────
def snr(sigma, vnorm, xi):
    return sigma**2 * vnorm**2 / (sigma**2 * vnorm**2 + xi**2 + 1e-30)


def th_rho(s, d, N):
    """Prop 2: on-manifold fraction."""
    return (1 + (N + 1) * s) / (d + (N + 1) * s)


def th_mean_cos(s, d, N):
    """Expected cosine of Δθ with gradient direction (approx for large N)."""
    rho = th_rho(s, d, N)
    # Mean update has component α·σ·||v||/σ_R in gradient direction
    # cos ≈ sqrt(rho) as a rough upper bound; exact from theory is more involved
    return rho  # return rho as proxy (equals fraction of power in gradient dir)


# ── MC helper ──────────────────────────────────────────────────────────────────
def measure_linear(sigma, d, N, xi, vnorm, nmc=2000):
    """
    Returns:
      rho_emp  — empirical on-manifold fraction E[||Δθ_∥||²]/E[||Δθ||²]
      cos_emp  — empirical mean cosine(Δθ, -v)
    """
    alpha = sigma / 2.0
    v     = torch.zeros(d, device=DEVICE); v[0] = vnorm  # gradient direction
    v_hat = v / (v.norm() + 1e-30)
    chunk = safe_chunk(d, N)

    sum_par, sum_tot, sum_cos, cnt = 0.0, 0.0, 0.0, 0
    while cnt < nmc:
        b   = min(chunk, nmc - cnt)
        eps = torch.randn(b, N, d, device=DEVICE)
        # Linear rewards: R_i = -sigma * v^T eps_i (+ noise)
        R   = -sigma * (eps @ v)                              # (b, N)
        if xi > 0:
            R = R + xi * torch.randn(b, N, device=DEVICE)
        Z   = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
        dth = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)   # (b, d)

        proj    = dth @ v_hat                                  # (b,)  parallel component scalar
        par_vec = proj.unsqueeze(-1) * v_hat.unsqueeze(0)    # (b, d) parallel component
        sum_par += (par_vec ** 2).sum(-1).sum().item()
        sum_tot += (dth ** 2).sum(-1).sum().item()
        # cosine with negative gradient direction (-v is the minimization direction)
        sum_cos += (-proj / (dth.norm(dim=-1) + 1e-9)).sum().item()
        cnt     += b
        del eps, R, Z, dth, par_vec; gc()

    rho_emp = (sum_par / nmc) / (sum_tot / nmc + 1e-30)
    cos_emp = sum_cos / nmc
    return rho_emp, cos_emp


# ══════════════════════════════════════════════════════════════════════════════
# Exp 2a: ρ vs ξ (reward noise / SNR sweep)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 2a: ρ vs reward noise ξ", flush=True)
print("="*60, flush=True)
t0 = time.time()

XI_VALS  = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
D_VALS_A = [100, 500, 1000]
N_FIXED  = 50
VNORM    = 1.0
NMC_2A   = 2000

rows_2a = []
for d in D_VALS_A:
    for xi in XI_VALS:
        rho_e, cos_e = measure_linear(SIGMA, d, N_FIXED, xi, VNORM, NMC_2A)
        s     = snr(SIGMA, VNORM, xi)
        rho_t = th_rho(s, d, N_FIXED)
        rows_2a.append(dict(d=d, xi=xi, snr=s, rho_emp=rho_e, rho_theory=rho_t, cos_emp=cos_e))
        print(f"  d={d:5d} xi={xi:.2f} s={s:.4f}: rho emp={rho_e:.4f} th={rho_t:.4f}  cos={cos_e:.4f}", flush=True)

df_2a = pd.DataFrame(rows_2a)
print(f"  2a done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
cmap_d = plt.cm.viridis(np.linspace(0.15, 0.85, len(D_VALS_A)))
for ai, d in enumerate(D_VALS_A):
    sub = df_2a[df_2a.d == d]
    axes[0].semilogx(sub.xi + 1e-3, sub.rho_emp,    'o-', color=cmap_d[ai], ms=7, lw=2,  label=f'd={d} emp')
    axes[0].semilogx(sub.xi + 1e-3, sub.rho_theory, 's--', color=cmap_d[ai], ms=5, lw=1.5, alpha=0.75)
    axes[1].semilogx(sub.snr + 1e-6, sub.rho_emp,   'o-', color=cmap_d[ai], ms=7, lw=2,  label=f'd={d} emp')
    axes[1].semilogx(sub.snr + 1e-6, sub.rho_theory,'s--', color=cmap_d[ai], ms=5, lw=1.5, alpha=0.75)
    axes[2].semilogx(sub.xi + 1e-3, sub.cos_emp,    'o-', color=cmap_d[ai], ms=7, lw=2,  label=f'd={d}')
axes[0].set_xlabel('ξ (reward noise)'); axes[0].set_ylabel('ρ (on-manifold fraction)')
axes[0].set_title(f'ρ vs ξ  (N={N_FIXED})'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('SNR s'); axes[1].set_ylabel('ρ')
axes[1].set_title('Prop 2: ρ vs SNR  (solid=emp, dash=theory)'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
axes[2].set_xlabel('ξ'); axes[2].set_ylabel('Mean cosine(Δθ, -∇)')
axes[2].set_title('Alignment with gradient'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
plt.suptitle('Exp 2a: On-manifold fraction ρ vs reward noise', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp2a_rho_vs_xi')
print("  → saved exp2a_rho_vs_xi.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 2b: ρ vs N (population size)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 2b: ρ vs N", flush=True)
print("="*60, flush=True)
t0 = time.time()

N_VALS_B  = [5, 10, 30, 50, 100, 300, 500, 1000]
D_VALS_B  = [100, 500, 1000, 5000]
XI_FIXED  = 0.1
NMC_2B    = 2000

rows_2b = []
for d in D_VALS_B:
    for N in N_VALS_B:
        rho_e, cos_e = measure_linear(SIGMA, d, N, XI_FIXED, VNORM, NMC_2B)
        s     = snr(SIGMA, VNORM, XI_FIXED)
        rho_t = th_rho(s, d, N)
        rows_2b.append(dict(d=d, N=N, rho_emp=rho_e, rho_theory=rho_t, cos_emp=cos_e))
        print(f"  d={d:5d} N={N:4d}: rho emp={rho_e:.5f} th={rho_t:.5f}", flush=True)

df_2b = pd.DataFrame(rows_2b)
print(f"  2b done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
cmap_d2 = plt.cm.plasma(np.linspace(0.1, 0.85, len(D_VALS_B)))
for ai, d in enumerate(D_VALS_B):
    sub = df_2b[df_2b.d == d]
    axes[0].loglog(sub.N, sub.rho_emp,    'o-',  color=cmap_d2[ai], ms=7, lw=2, label=f'd={d} emp')
    axes[0].loglog(sub.N, sub.rho_theory, 's--', color=cmap_d2[ai], ms=5, lw=1.5, alpha=0.75)
    axes[1].loglog(sub.N, sub.rho_emp / sub.rho_theory, 'o-', color=cmap_d2[ai], ms=7, lw=2, label=f'd={d}')
axes[0].set_xlabel('N'); axes[0].set_ylabel('ρ')
axes[0].set_title(f'Exp 2b: ρ vs N  (ξ={XI_FIXED}, solid=emp, dash=theory)'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].axhline(1.0, color='k', ls='--', lw=1.5)
axes[1].set_xlabel('N'); axes[1].set_ylabel('ρ emp / ρ theory')
axes[1].set_title('Ratio (should be ≈ 1)'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
plt.suptitle('Exp 2b: On-manifold fraction ρ vs population size N', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp2b_rho_vs_N')
print("  → saved exp2b_rho_vs_N.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 2c: ρ vs d (dimensionality curse)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 2c: ρ vs d (curse of dimensionality)", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_VALS_C = [50, 100, 200, 500, 1000, 2000, 5000]
N_VALS_C = [50, 200, 500]
XI_VALS_C = [0.0, 0.1, 1.0]
NMC_2C   = 2000

rows_2c = []
for N in N_VALS_C:
    for xi in XI_VALS_C:
        for d in D_VALS_C:
            rho_e, _ = measure_linear(SIGMA, d, N, xi, VNORM, NMC_2C)
            s        = snr(SIGMA, VNORM, xi)
            rho_t    = th_rho(s, d, N)
            # theoretical N* for rho=0.5
            N_star   = (0.5 * d - 1) / (s * 0.5) - 1 if s > 0 else float('inf')
            rows_2c.append(dict(d=d, N=N, xi=xi, snr=s, rho_emp=rho_e, rho_theory=rho_t, N_star=N_star))
            print(f"  N={N:4d} xi={xi:.1f} d={d:5d}: rho emp={rho_e:.5f} th={rho_t:.5f}  N*={N_star:.0f}", flush=True)

df_2c = pd.DataFrame(rows_2c)
print(f"  2c done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
cmap_N2 = plt.cm.cool(np.linspace(0.1, 0.85, len(N_VALS_C)))

# panel 0: rho vs d for each N, xi=0.1
xi_plot = 0.1
for ni, N in enumerate(N_VALS_C):
    sub = df_2c[(df_2c.N == N) & (df_2c.xi == xi_plot)]
    axes[0].semilogx(sub.d, sub.rho_emp,    'o-',  color=cmap_N2[ni], ms=7, lw=2, label=f'N={N} emp')
    axes[0].semilogx(sub.d, sub.rho_theory, 's--', color=cmap_N2[ni], ms=5, lw=1.5, alpha=0.75)
axes[0].set_xlabel('d'); axes[0].set_ylabel('ρ')
axes[0].set_title(f'ρ vs d  (ξ={xi_plot})'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# panel 1: N* (population needed for rho>0.5) vs d
s_xi_vals = {xi: snr(SIGMA, VNORM, xi) for xi in [0.0, 0.1, 1.0]}
d_range   = np.logspace(np.log10(50), np.log10(5000), 100)
cmap_xi2  = plt.cm.autumn(np.linspace(0.1, 0.9, 3))
for ci, (xi, sv) in enumerate(s_xi_vals.items()):
    if sv > 1e-6:
        N_star_arr = (0.5 * d_range - 1) / (sv * 0.5) - 1
        axes[1].loglog(d_range, np.maximum(N_star_arr, 1), '-', color=cmap_xi2[ci], lw=2, label=f'ξ={xi}')
for N in N_VALS_C:
    axes[1].axhline(N, color='gray', ls=':', lw=1)
    axes[1].text(5200, N*1.05, f'N={N}', fontsize=7, color='gray')
axes[1].set_xlabel('d'); axes[1].set_ylabel('N* (for ρ > 0.5)')
axes[1].set_title('N* ≈ d/s — linear curse'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# panel 2: ρ vs d for each xi, N=200
N_plot = 200
for ci, xi in enumerate([0.0, 0.1, 1.0]):
    sub = df_2c[(df_2c.N == N_plot) & (df_2c.xi == xi)]
    if len(sub) == 0: continue
    axes[2].semilogx(sub.d, sub.rho_emp,    'o-',  color=cmap_xi2[ci], ms=7, lw=2, label=f'ξ={xi} emp')
    axes[2].semilogx(sub.d, sub.rho_theory, 's--', color=cmap_xi2[ci], ms=5, lw=1.5, alpha=0.75)
axes[2].set_xlabel('d'); axes[2].set_ylabel('ρ')
axes[2].set_title(f'ρ vs d for ξ values (N={N_plot})'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.suptitle('Exp 2c: Curse of dimensionality — ρ falls as d grows', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp2c_rho_vs_d')
print("  → saved exp2c_rho_vs_d.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 2e: Multi-step convergence on linear landscape
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 2e: Multi-step convergence on linear landscape", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_MS = 500; K_MS = 1; T_MS = 300; NREP_MS = 50
XI_MS_VALS = [0.0, 0.1, 1.0, 5.0]
N_MS_VALS  = [50, 200]

rows_2e = []
for N in N_MS_VALS:
    for xi in XI_MS_VALS:
        alpha = SIGMA / 2.0
        v = torch.zeros(D_MS, device=DEVICE); v[0] = VNORM
        # Run NREP trajectories: start from theta=3*ones
        theta = 3.0 * torch.ones(NREP_MS, D_MS, device=DEVICE)
        theta_star = torch.zeros(D_MS, device=DEVICE)
        for t in range(1, T_MS + 1):
            eps = torch.randn(NREP_MS, N, D_MS, device=DEVICE)
            # R = -sigma * v^T (theta + sigma*eps)  → rewards
            perturbed = theta.unsqueeze(1) + SIGMA * eps  # (NREP, N, d)
            R = -(perturbed @ v)                           # (NREP, N)
            if xi > 0: R = R + xi * torch.randn(NREP_MS, N, device=DEVICE)
            Z   = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
            dth = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)  # (NREP, d)
            theta = theta + dth
            if t % 30 == 0 or t == T_MS:
                dist = (theta - theta_star.unsqueeze(0)).norm(dim=-1).mean().item()
                rows_2e.append(dict(N=N, xi=xi, T=t, mean_dist=dist))
            del eps, perturbed, R, Z, dth; gc()
        del theta; gc()
        print(f"  N={N} xi={xi}: final dist={rows_2e[-1]['mean_dist']:.4f}", flush=True)

df_2e = pd.DataFrame(rows_2e)
print(f"  2e done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
cmap_xi3 = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(XI_MS_VALS)))
for ni, N in enumerate(N_MS_VALS):
    ax = axes[ni]
    for ci, xi in enumerate(XI_MS_VALS):
        sub = df_2e[(df_2e.N == N) & (df_2e.xi == xi)]
        ax.semilogy(sub['T'], sub.mean_dist, 'o-', color=cmap_xi3[ci], ms=5, lw=2, label=f'ξ={xi}')
    ax.set_xlabel('Step T'); ax.set_ylabel('Mean ||θ_t - θ*||')
    ax.set_title(f'N={N}: Convergence vs noise ξ'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle(f'Exp 2e: Multi-step convergence on linear landscape (d={D_MS})', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp2e_multistep_linear')
print("  → saved exp2e_multistep_linear.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Save all data
# ══════════════════════════════════════════════════════════════════════════════
all_data = dict(exp2a=df_2a, exp2b=df_2b, exp2c=df_2c, exp2e=df_2e,
                params=dict(sigma=SIGMA, vnorm=VNORM))
with open(DATA / 'exp2_linear.pkl', 'wb') as f:
    pickle.dump(all_data, f)

df_2a.to_csv(DATA / 'exp2a_rho_vs_xi.csv', index=False)
df_2b.to_csv(DATA / 'exp2b_rho_vs_N.csv', index=False)
df_2c.to_csv(DATA / 'exp2c_rho_vs_d.csv', index=False)
df_2e.to_csv(DATA / 'exp2e_multistep.csv', index=False)
print(f"\nSaved pkl + csv to {DATA}", flush=True)
print(f"\nExp 2 total time: {time.time()-T0_TOTAL:.1f}s", flush=True)
print("Exp 2 DONE", flush=True)
