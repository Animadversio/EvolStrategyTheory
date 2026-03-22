#!/usr/bin/env python3
"""Exp 3: Quadratic landscape — σ_R formula and convergence (Prop 3).

Landscape: R(θ) = -½(θ-θ*)^T H (θ-θ*)  where H = Q^T Λ Q
ZScoreES gradient: E[Δθ] ≈ α·σ·v / σ_R   where v = ∇f at current θ

Prop 3:
  σ_R² = σ²||v||² + ½σ⁴·tr(Q²) + ξ²
  where v = H(θ-θ*) is the gradient, tr(Q²) = Σ λ_i²

Sub-experiments:
  3a: σ_R theory vs empirical across spectra and noise levels
  3b: Mean update magnitude ‖E[Δθ]‖ theory vs empirical
  3c: Multi-step convergence trajectories (varying d, N, ξ)
  3d: Scaling with d — convergence rate vs theory
  3e: N scaling — compare ZScoreES convergence speed vs N
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


# ── landscape helpers ──────────────────────────────────────────────────────────
def make_rot(d, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R)); signs[signs == 0] = 1.0
    return (Q * signs.unsqueeze(0)).to(DEVICE)


def make_eigs(d, k, spectrum, lam_max=5.0, lam_min=0.5, flat=1e-4, beta=1.0):
    eigs = torch.full((d,), flat, dtype=torch.float32)
    if spectrum == 'uniform':
        eigs[:k] = lam_max
    elif spectrum == 'powerlaw':
        eigs[:k] = torch.tensor([lam_max * (i+1)**(-beta) for i in range(k)], dtype=torch.float32)
    elif spectrum == 'range':
        eigs[:k] = torch.logspace(np.log10(lam_max), np.log10(lam_min), k, dtype=torch.float32)
    return eigs.to(DEVICE)


# ── theory formulae ────────────────────────────────────────────────────────────
def th_sigmaR_sq(sigma, vnorm, trQ2, xi):
    return sigma**2 * vnorm**2 + 0.5 * sigma**4 * trQ2 + xi**2


def th_mean_update(alpha, sigma, vnorm, sigmaR):
    """||E[Δθ]|| = α·σ·||v|| / σ_R"""
    return alpha * sigma * vnorm / (sigmaR + 1e-30)


def compute_v_and_sigmaR(Rot, eigs, theta, theta_star, sigma, xi):
    """Analytically compute v = H(θ-θ*), trQ2, σ_R."""
    dx   = theta - theta_star
    y    = Rot @ dx
    v    = Rot.T @ (eigs * y)             # gradient direction
    vnorm  = v.norm().item()
    trQ2   = (eigs ** 2).sum().item()
    sigmaR = np.sqrt(th_sigmaR_sq(sigma, vnorm, trQ2, xi))
    return v, vnorm, trQ2, sigmaR


# ── MC measurement of σ_R and mean update ─────────────────────────────────────
def measure_quad_step(Rot, eigs, theta, theta_star, sigma, xi, N, nmc=2000):
    """
    Returns:
      sigmaR_emp — empirical reward std
      mean_up_emp — empirical ||mean update||
    """
    alpha = sigma / 2.0
    d     = theta.shape[0]
    chunk = safe_chunk(d, N)
    sum_R2, sum_R, sum_dth, cnt = 0.0, 0.0, torch.zeros(d, device=DEVICE), 0
    while cnt < nmc:
        b   = min(chunk, nmc - cnt)
        eps = torch.randn(b, N, d, device=DEVICE)
        # Perturbed positions: shape (b, N, d)
        dx  = theta.unsqueeze(0).unsqueeze(0) + sigma * eps - theta_star.unsqueeze(0).unsqueeze(0)
        y   = dx @ Rot.T          # (b, N, d) in eigenbasis
        R   = -0.5 * (eigs * y * y).sum(-1)  # (b, N) quadratic rewards
        if xi > 0:
            R = R + xi * torch.randn(b, N, device=DEVICE)
        Z   = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
        dth = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)  # (b, d)
        # accumulate for σ_R (use ALL rewards from all perturbations)
        R_flat = R.reshape(-1)
        sum_R2  += (R_flat ** 2).sum().item()
        sum_R   += R_flat.sum().item()
        sum_dth += dth.sum(0)
        cnt     += b
        del eps, dx, y, R, Z, dth; gc()
    total_samples = nmc * N
    var_R    = sum_R2 / total_samples - (sum_R / total_samples) ** 2
    sigmaR_emp   = np.sqrt(max(var_R, 0))
    mean_up_emp  = (sum_dth / nmc).norm().item()
    return sigmaR_emp, mean_up_emp


# ══════════════════════════════════════════════════════════════════════════════
# Exp 3a: σ_R theory vs empirical across spectra and ξ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 3a: σ_R theory vs empirical", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_3A = 500; K_3A = 20; N_3A = 50; NMC_3A = 2000
SPECTRA  = [('uniform', 'Uniform λ=5'), ('powerlaw', 'Power-law β=1'), ('range', 'Range [0.5,5]')]
XI_VALS_3 = [0.0, 0.1, 0.5, 1.0, 5.0]
Rot_3a    = make_rot(D_3A, seed=42)
theta_3a  = 3.0 * torch.randn(D_3A, device=DEVICE)
ts_3a     = torch.zeros(D_3A, device=DEVICE)

rows_3a = []
for sp_tag, sp_label in SPECTRA:
    eigs = make_eigs(D_3A, K_3A, sp_tag, lam_max=5.0, lam_min=0.5, flat=1e-4)
    _, vnorm, trQ2, _ = compute_v_and_sigmaR(Rot_3a, eigs, theta_3a, ts_3a, SIGMA, 0.0)
    for xi in XI_VALS_3:
        sigmaR_th  = np.sqrt(th_sigmaR_sq(SIGMA, vnorm, trQ2, xi))
        sigmaR_emp, mean_up_emp = measure_quad_step(Rot_3a, eigs, theta_3a, ts_3a, SIGMA, xi, N_3A, NMC_3A)
        mean_up_th = th_mean_update(ALPHA, SIGMA, vnorm, sigmaR_th)
        rows_3a.append(dict(spectrum=sp_label, xi=xi, vnorm=vnorm, trQ2=trQ2,
                            sigmaR_th=sigmaR_th, sigmaR_emp=sigmaR_emp,
                            mean_up_th=mean_up_th, mean_up_emp=mean_up_emp,
                            ratio_sigmaR=sigmaR_emp/sigmaR_th,
                            ratio_up=mean_up_emp/(mean_up_th+1e-30)))
        print(f"  {sp_label:20s} xi={xi:.1f}: σ_R th={sigmaR_th:.4f} emp={sigmaR_emp:.4f}  "
              f"||E[Δθ]|| th={mean_up_th:.5f} emp={mean_up_emp:.5f}", flush=True)
    del eigs; gc()

df_3a = pd.DataFrame(rows_3a)
print(f"  3a done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
cmap_sp = plt.cm.tab10(np.linspace(0, 0.3, len(SPECTRA)))
for si, (sp_tag, sp_label) in enumerate(SPECTRA):
    sub = df_3a[df_3a.spectrum == sp_label]
    axes[0].semilogy(sub.xi + 1e-3, sub.sigmaR_th,  '--', color=cmap_sp[si], lw=2, label=f'{sp_label} th')
    axes[0].semilogy(sub.xi + 1e-3, sub.sigmaR_emp, 'o',  color=cmap_sp[si], ms=8)
    axes[1].semilogy(sub.xi + 1e-3, sub.mean_up_th,  '--', color=cmap_sp[si], lw=2, label=f'{sp_label} th')
    axes[1].semilogy(sub.xi + 1e-3, sub.mean_up_emp, 'o',  color=cmap_sp[si], ms=8)
    axes[2].semilogx(sub.xi + 1e-3, sub.ratio_sigmaR, 'o-', color=cmap_sp[si], ms=7, lw=2, label=sp_label)
axes[0].set_xlabel('ξ'); axes[0].set_ylabel('σ_R')
axes[0].set_title(f'Prop 3: σ_R  (d={D_3A}, k={K_3A})'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('ξ'); axes[1].set_ylabel('||E[Δθ]||')
axes[1].set_title('Mean update magnitude'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
axes[2].axhline(1.0, color='k', ls='--', lw=1.5)
axes[2].set_xlabel('ξ'); axes[2].set_ylabel('σ_R emp / σ_R theory')
axes[2].set_title('Ratio σ_R emp/theory (≈ 1)'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
plt.suptitle('Exp 3a: σ_R and mean update — Prop 3 validation', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp3a_sigmaR')
print("  → saved exp3a_sigmaR.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 3c: Multi-step convergence (varying d, N, ξ)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 3c: Multi-step convergence on quadratic landscape", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_MS_VALS = [200, 500, 1000]
N_MS_VALS = [50, 200]
XI_MS     = [0.0, 0.1, 1.0]
K_MS = 20; T_MS = 300; NREP_MS = 30

rows_3c = []
for d in D_MS_VALS:
    Rot  = make_rot(d, seed=42)
    eigs = make_eigs(d, K_MS, 'uniform', lam_max=5.0, flat=1e-4)
    trQ2 = (eigs ** 2).sum().item()
    ts   = torch.zeros(d, device=DEVICE)
    for N in N_MS_VALS:
        for xi in XI_MS:
            alpha = SIGMA / 2.0
            # NREP independent trajectories
            theta = 5.0 * torch.ones(NREP_MS, d, device=DEVICE)
            dist_traj, sigmaR_th_traj, eff_step_traj = [], [], []
            for t in range(1, T_MS + 1):
                # analytical σ_R at current mean theta (use mean of trajectories)
                theta_mean = theta.mean(0)
                _, vnorm_t, _, sigmaR_t = compute_v_and_sigmaR(Rot, eigs, theta_mean, ts, SIGMA, xi)
                eff_step_t = th_mean_update(ALPHA, SIGMA, vnorm_t, sigmaR_t)

                # empirical step
                eps  = torch.randn(NREP_MS, N, d, device=DEVICE)
                dx   = theta.unsqueeze(1) + SIGMA * eps - ts.unsqueeze(0).unsqueeze(0)  # (R, N, d)
                y    = dx @ Rot.T
                R    = -0.5 * (eigs * y * y).sum(-1)   # (NREP, N)
                if xi > 0: R = R + xi * torch.randn(NREP_MS, N, device=DEVICE)
                Z    = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
                dth  = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)
                theta = theta + dth

                if t % 30 == 0 or t == T_MS:
                    dist = (theta - ts.unsqueeze(0)).norm(dim=-1).mean().item()
                    dist_traj.append((t, dist, sigmaR_t, eff_step_t))
                del eps, dx, y, R, Z, dth; gc()

            for t_val, dist, sR, es in dist_traj:
                rows_3c.append(dict(d=d, N=N, xi=xi, T=t_val, dist=dist, sigmaR_th=sR, eff_step=es))
            print(f"  d={d} N={N} xi={xi}: final dist={dist_traj[-1][1]:.4f}", flush=True)
            del theta; gc()
    del Rot, eigs; gc()

df_3c = pd.DataFrame(rows_3c)
print(f"  3c done in {time.time()-t0:.1f}s", flush=True)

# Plot: 3×2 grid — rows=d, cols=N
fig, axes = plt.subplots(len(D_MS_VALS), len(N_MS_VALS), figsize=(12, 12))
cmap_xi4 = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(XI_MS)))
for ri, d in enumerate(D_MS_VALS):
    for ci, N in enumerate(N_MS_VALS):
        ax = axes[ri][ci]
        for xii, xi in enumerate(XI_MS):
            sub = df_3c[(df_3c.d == d) & (df_3c.N == N) & (df_3c.xi == xi)]
            ax.semilogy(sub['T'], sub.dist, 'o-', color=cmap_xi4[xii], ms=5, lw=2, label=f'ξ={xi}')
        ax.set_xlabel('Step T'); ax.set_ylabel('||θ - θ*||')
        ax.set_title(f'd={d}, N={N}'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle('Exp 3c: Multi-step convergence on quadratic landscape', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp3c_multistep_quad')
print("  → saved exp3c_multistep_quad.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 3d: Scaling with d — convergence rate vs dimensionality
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 3d: Convergence rate vs d", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_SCALE = [100, 300, 500, 1000, 2000, 5000]
N_D_FIXED = 100; K_D = 20; T_D = 150; NREP_D = 20; XI_D = 0.1

rows_3d = []
for d in D_SCALE:
    Rot  = make_rot(d, seed=7)
    eigs = make_eigs(d, K_D, 'uniform', lam_max=5.0, flat=1e-4)
    ts   = torch.zeros(d, device=DEVICE)
    theta = 5.0 * torch.ones(NREP_D, d, device=DEVICE)
    alpha = SIGMA / 2.0

    for t in range(1, T_D + 1):
        eps  = torch.randn(NREP_D, N_D_FIXED, d, device=DEVICE)
        dx   = theta.unsqueeze(1) + SIGMA * eps - ts.unsqueeze(0).unsqueeze(0)
        y    = dx @ Rot.T
        R    = -0.5 * (eigs * y * y).sum(-1)
        R    = R + XI_D * torch.randn(NREP_D, N_D_FIXED, device=DEVICE)
        Z    = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
        dth  = (alpha / N_D_FIXED) * (Z.unsqueeze(-1) * eps).sum(1)
        theta = theta + dth
        if t in [1, 10, 30, 50, 100, T_D]:
            dist = (theta - ts.unsqueeze(0)).norm(dim=-1).mean().item()
            # theoretical: initial eff_step ≈ α·σ·||v_0||/σ_R_0
            theta_mean = theta.mean(0)
            _, vnorm_t, trQ2_t, sigmaR_t = compute_v_and_sigmaR(Rot, eigs, theta_mean, ts, SIGMA, XI_D)
            rows_3d.append(dict(d=d, T=t, dist=dist, sigmaR_th=sigmaR_t,
                                eff_step=th_mean_update(ALPHA, SIGMA, vnorm_t, sigmaR_t)))
        del eps, dx, y, R, Z, dth; gc()

    print(f"  d={d:5d}: final dist={rows_3d[-1]['dist']:.4f}  "
          f"init sigmaR_th={rows_3d[0]['sigmaR_th']:.4f}", flush=True)
    del theta, Rot, eigs; gc()

df_3d = pd.DataFrame(rows_3d)
print(f"  3d done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
cmap_d3 = plt.cm.viridis(np.linspace(0.1, 0.9, len(D_SCALE)))
for di, d in enumerate(D_SCALE):
    sub = df_3d[df_3d.d == d]
    axes[0].semilogy(sub['T'], sub.dist, 'o-', color=cmap_d3[di], ms=5, lw=2, label=f'd={d}')

# At final step: dist vs d
final = df_3d[df_3d['T'] == T_D]
axes[1].loglog(final.d, final.dist, 'o-', ms=8, lw=2, color='steelblue', label='final dist')
axes[1].set_xlabel('d'); axes[1].set_ylabel(f'||θ_{T_D} - θ*||')
axes[1].set_title(f'Final distance vs d  (T={T_D}, N={N_D_FIXED})'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Effective step size at T=1 vs d
init = df_3d[df_3d['T'] == 1]
axes[2].loglog(init.d, init.eff_step, 'o-', ms=8, lw=2, color='darkorange', label='eff step (th)')
axes[2].set_xlabel('d'); axes[2].set_ylabel('γ = α·σ·||v||/σ_R')
axes[2].set_title('Initial eff. step size vs d'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

axes[0].set_xlabel('Step T'); axes[0].set_ylabel('||θ - θ*||')
axes[0].set_title(f'Convergence vs d (N={N_D_FIXED}, k={K_D})'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
plt.suptitle('Exp 3d: Convergence rate scaling with dimensionality d', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp3d_d_scaling')
print("  → saved exp3d_d_scaling.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 3e: N scaling — convergence speed vs population size
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("Exp 3e: Convergence speed vs N", flush=True)
print("="*60, flush=True)
t0 = time.time()

D_3E = 500; K_3E = 20; T_3E = 300; NREP_3E = 30; XI_3E = 0.1
N_SCALE = [10, 30, 50, 100, 200, 500]

rows_3e = []
Rot_e  = make_rot(D_3E, seed=13)
eigs_e = make_eigs(D_3E, K_3E, 'uniform', lam_max=5.0, flat=1e-4)
ts_e   = torch.zeros(D_3E, device=DEVICE)

for N in N_SCALE:
    alpha = SIGMA / 2.0
    theta = 5.0 * torch.ones(NREP_3E, D_3E, device=DEVICE)
    for t in range(1, T_3E + 1):
        eps  = torch.randn(NREP_3E, N, D_3E, device=DEVICE)
        dx   = theta.unsqueeze(1) + SIGMA * eps - ts_e.unsqueeze(0).unsqueeze(0)
        y    = dx @ Rot_e.T
        R    = -0.5 * (eigs_e * y * y).sum(-1)
        R    = R + XI_3E * torch.randn(NREP_3E, N, device=DEVICE)
        Z    = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True, correction=0) + 1e-9)
        dth  = (alpha / N) * (Z.unsqueeze(-1) * eps).sum(1)
        theta = theta + dth
        if t % 30 == 0 or t == T_3E:
            dist = (theta - ts_e.unsqueeze(0)).norm(dim=-1).mean().item()
            rows_3e.append(dict(N=N, T=t, dist=dist))
        del eps, dx, y, R, Z, dth; gc()
    print(f"  N={N:4d}: final dist={rows_3e[-1]['dist']:.4f}", flush=True)
    del theta; gc()

del Rot_e, eigs_e; gc()
df_3e = pd.DataFrame(rows_3e)
print(f"  3e done in {time.time()-t0:.1f}s", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
cmap_N3 = plt.cm.plasma(np.linspace(0.1, 0.9, len(N_SCALE)))
for ni, N in enumerate(N_SCALE):
    sub = df_3e[df_3e.N == N]
    axes[0].semilogy(sub['T'], sub.dist, 'o-', color=cmap_N3[ni], ms=5, lw=2, label=f'N={N}')
axes[0].set_xlabel('Step T'); axes[0].set_ylabel('||θ - θ*||')
axes[0].set_title(f'Convergence vs N  (d={D_3E}, ξ={XI_3E})'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

final_e = df_3e[df_3e['T'] == T_3E]
axes[1].loglog(final_e.N, final_e.dist, 'o-', ms=8, lw=2, color='darkgreen', label='final dist')
axes[1].set_xlabel('N'); axes[1].set_ylabel(f'||θ_{T_3E} - θ*||')
axes[1].set_title(f'Final distance vs N (T={T_3E})'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle(f'Exp 3e: Convergence vs population size N (d={D_3E})', fontsize=13)
plt.tight_layout()
save_fig(fig, 'exp3e_N_scaling')
print("  → saved exp3e_N_scaling.png", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Save all data
# ══════════════════════════════════════════════════════════════════════════════
all_data = dict(exp3a=df_3a, exp3c=df_3c, exp3d=df_3d, exp3e=df_3e,
                params=dict(sigma=SIGMA, alpha=ALPHA))
with open(DATA / 'exp3_quadratic.pkl', 'wb') as f:
    pickle.dump(all_data, f)

df_3a.to_csv(DATA / 'exp3a_sigmaR.csv', index=False)
df_3c.to_csv(DATA / 'exp3c_multistep.csv', index=False)
df_3d.to_csv(DATA / 'exp3d_d_scaling.csv', index=False)
df_3e.to_csv(DATA / 'exp3e_N_scaling.csv', index=False)
print(f"\nSaved pkl + csv to {DATA}", flush=True)
print(f"\nExp 3 total time: {time.time()-T0_TOTAL:.1f}s", flush=True)
print("Exp 3 DONE", flush=True)
