#!/usr/bin/env python3
"""Exp 1: Flat landscape — drift variance validation (Prop 1).

Prop 1:  E[||Δθ||²] = α²d/N = σ²d/(4N)   per step
         E[||θ_T - θ_0||²] = σ²Td/(4N)   cumulative (random walk)

Sub-experiments:
  1a: One-step drift vs N and d (2D grid), emp vs theory
  1b: σ scaling — does α²d/N hold across σ values?
  1c: Cumulative drift vs T (multi-step random walk)
  1d: Summary ratio table emp/theory
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

# ── paths ─────────────────────────────────────────────────────────────────────
STORE = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2")
FIGS  = STORE / "figures"
DATA  = STORE / "data"
for p in [FIGS, DATA]: p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

T0_TOTAL = time.time()


def gc():
    if DEVICE.type == 'cuda':
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()


def safe_chunk(d, N, gb=1.5):
    """Max batch size s.t. (b, N, d) float32 tensor < gb GB."""
    return max(1, int(gb * 1e9 / (4 * N * d)))


# ── theory formulae ────────────────────────────────────────────────────────────
def th_step(sigma, N, d):
    """E[||Δθ||²] = α²d/N = σ²d/(4N)."""
    return (sigma / 2)**2 * d / N


def th_cumul(sigma, T, d, N):
    """E[||θ_T - θ_0||²] = σ²Td/(4N)."""
    return sigma**2 * T * d / (4 * N)


# ── MC helper ──────────────────────────────────────────────────────────────────
def measure_flat_step(sigma, N, d, nmc=2000):
    """Empirical E[||Δθ||²] on flat landscape (rewards = pure noise)."""
    alpha = sigma / 2.0
    chunk = safe_chunk(d, N)
    total, cnt = 0.0, 0
    while cnt < nmc:
        b = min(chunk, nmc - cnt)
        eps = torch.randn(b, N, d, device=DEVICE)
        R   = torch.randn(b, N, device=DEVICE)          # flat: rewards are noise
        Z   = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
        dth = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)  # (b, d)
        total += (dth ** 2).sum(-1).sum().item()
        cnt   += b
        del eps, R, Z, dth; gc()
    return total / nmc


# ══════════════════════════════════════════════════════════════════════════════
# Exp 1a: One-step drift vs N and d
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 1a: One-step drift E[||Δθ||²] vs N and d", flush=True)
print("="*60, flush=True)
t0 = time.time()

SIGMA_DEFAULT = 0.1
D_VALS = [100, 500, 1000, 5000]
N_VALS = [10, 30, 100, 300, 1000]
NMC_1A = 2000

rows_1a = []
for d in D_VALS:
    for N in N_VALS:
        emp  = measure_flat_step(SIGMA_DEFAULT, N, d, NMC_1A)
        th   = th_step(SIGMA_DEFAULT, N, d)
        ratio = emp / th
        rows_1a.append(dict(d=d, N=N, sigma=SIGMA_DEFAULT, emp=emp, theory=th, ratio=ratio))
        print(f"  d={d:5d} N={N:4d}:  emp={emp:.4e}  theory={th:.4e}  ratio={ratio:.4f}", flush=True)

df_1a = pd.DataFrame(rows_1a)
print(f"  1a done in {time.time()-t0:.1f}s", flush=True)

# Plot: rows=d, cols=N, emp vs theory
fig, axes = plt.subplots(1, len(D_VALS), figsize=(16, 4.5), sharey=False)
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(D_VALS)))
for ai, d in enumerate(D_VALS):
    sub  = df_1a[df_1a.d == d]
    axes[ai].loglog(sub.N, sub.emp,    'o-', color=cmap[ai], ms=7, lw=2,   label='empirical')
    axes[ai].loglog(sub.N, sub.theory, 's--', color='k',      ms=5, lw=1.5, label='theory σ²d/4N')
    axes[ai].set_xlabel('N (population size)'); axes[ai].set_ylabel(r'$\mathbb{E}[||\Delta\theta||^2]$')
    axes[ai].set_title(f'd = {d}'); axes[ai].legend(fontsize=8); axes[ai].grid(True, alpha=0.3)
plt.suptitle('Exp 1a: One-step drift variance vs N (flat landscape)', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp1a_drift_vs_N')
print("  → saved exp1a_drift_vs_N.png", flush=True)

# Ratio heatmap — seaborn annotated
pivot = df_1a.pivot(index='d', columns='N', values='ratio')
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
            vmin=0.9, vmax=1.1, linewidths=0.5,
            annot_kws={'size': 9}, ax=ax,
            cbar_kws={'label': 'emp / theory'})
ax.set_xlabel('N'); ax.set_ylabel('d')
ax.set_title('Exp 1a: Ratio emp / theory  (should be ≈ 1.0)')
plt.tight_layout()
save_fig(fig, 'exp1a_ratio_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
# Exp 1b: σ scaling — α²d/N holds for all σ?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 1b: σ scaling", flush=True)
print("="*60, flush=True)
t0 = time.time()

SIGMA_VALS = [0.05, 0.1, 0.2, 0.5, 1.0]
D_SIG, N_SIG = 500, 50
NMC_1B = 2000

rows_1b = []
for sigma in SIGMA_VALS:
    emp  = measure_flat_step(sigma, N_SIG, D_SIG, NMC_1B)
    th   = th_step(sigma, N_SIG, D_SIG)
    ratio = emp / th
    rows_1b.append(dict(sigma=sigma, emp=emp, theory=th, ratio=ratio))
    print(f"  sigma={sigma:.2f}: emp={emp:.4e}  theory={th:.4e}  ratio={ratio:.4f}", flush=True)

df_1b = pd.DataFrame(rows_1b)
print(f"  1b done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].loglog(df_1b.sigma, df_1b.emp,    'o-', ms=8, lw=2, label='empirical')
axes[0].loglog(df_1b.sigma, df_1b.theory, 's--', ms=6, lw=1.5, color='k', label='theory σ²d/4N')
axes[0].set_xlabel('σ'); axes[0].set_ylabel(r'$\mathbb{E}[||\Delta\theta||^2]$')
axes[0].set_title(f'Exp 1b: σ scaling (d={D_SIG}, N={N_SIG})'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].semilogx(df_1b.sigma, df_1b.ratio, 'o-', ms=8, lw=2)
axes[1].axhline(1.0, color='k', ls='--', lw=1.5)
axes[1].set_ylim([0.85, 1.15]); axes[1].set_xlabel('σ'); axes[1].set_ylabel('emp / theory')
axes[1].set_title('Ratio (should be ≈ 1)'); axes[1].grid(True, alpha=0.3)
plt.suptitle('Exp 1b: σ scaling of per-step drift', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp1b_sigma_scaling')
print("  → saved exp1b_sigma_scaling.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 1c: Cumulative drift vs T (multi-step random walk)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 1c: Cumulative drift vs T", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_MULTI = [100, 500, 1000]
N_MULTI = [10, 50, 200]
SIGMA_M  = 0.1
T_STEPS  = 200
NREP     = 300   # independent trajectories

rows_1c = []
for d in D_MULTI:
    for N in N_MULTI:
        alpha = SIGMA_M / 2.0
        # Run NREP trajectories simultaneously
        cum = torch.zeros(NREP, d, device=DEVICE)
        dist_sq_traj = []
        for t in range(1, T_STEPS + 1):
            eps = torch.randn(NREP, N, d, device=DEVICE)
            R   = torch.randn(NREP, N, device=DEVICE)
            Z   = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
            dth = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)
            cum = cum + dth
            if t % 20 == 0 or t == T_STEPS:
                dist_sq_traj.append((t, (cum**2).sum(1).mean().item()))
            del eps, R, Z, dth; gc()
        del cum; gc()
        for t_val, dsq in dist_sq_traj:
            th_val = th_cumul(SIGMA_M, t_val, d, N)
            rows_1c.append(dict(d=d, N=N, T=t_val, emp=dsq, theory=th_val, ratio=dsq/th_val))
        print(f"  d={d} N={N}: final_emp={dist_sq_traj[-1][1]:.2f}  theory={th_cumul(SIGMA_M,T_STEPS,d,N):.2f}", flush=True)

df_1c = pd.DataFrame(rows_1c)
print(f"  1c done in {time.time()-t0:.1f}s", flush=True)

# Plot: one panel per d, curves colored by N
fig, axes = plt.subplots(1, len(D_MULTI), figsize=(15, 4.5))
cmap_N = plt.cm.plasma(np.linspace(0.1, 0.85, len(N_MULTI)))
for ai, d in enumerate(D_MULTI):
    for ni, N in enumerate(N_MULTI):
        sub = df_1c[(df_1c.d == d) & (df_1c.N == N)]
        axes[ai].plot(sub['T'], sub.emp,    'o-', color=cmap_N[ni], ms=5, lw=1.8, label=f'N={N} emp')
        axes[ai].plot(sub['T'], sub.theory, '--',  color=cmap_N[ni], lw=1.5, alpha=0.7, label=f'N={N} th')
    axes[ai].set_xlabel('Step T'); axes[ai].set_ylabel(r'$\mathbb{E}[||\theta_T - \theta_0||^2]$')
    axes[ai].set_title(f'd = {d}'); axes[ai].legend(fontsize=7); axes[ai].grid(True, alpha=0.3)
plt.suptitle(f'Exp 1c: Cumulative drift vs T  (σ={SIGMA_M}, random walk prediction)', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp1c_cumulative_drift')
print("  → saved exp1c_cumulative_drift.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Save all data
# ══════════════════════════════════════════════════════════════════════════════
all_data = dict(exp1a=df_1a, exp1b=df_1b, exp1c=df_1c,
                params=dict(sigma_default=SIGMA_DEFAULT, nmc_1a=NMC_1A, nmc_1b=NMC_1B,
                            nrep_1c=NREP, T_steps=T_STEPS))
with open(DATA / 'exp1_flat.pkl', 'wb') as f:
    pickle.dump(all_data, f)

df_1a.to_csv(DATA / 'exp1a_drift_vs_Nd.csv', index=False)
df_1b.to_csv(DATA / 'exp1b_sigma_scaling.csv', index=False)
df_1c.to_csv(DATA / 'exp1c_cumulative.csv', index=False)
print(f"\nSaved pkl + csv to {DATA}", flush=True)
print(f"\nExp 1 total time: {time.time()-T0_TOTAL:.1f}s", flush=True)
print("Exp 1 DONE", flush=True)
