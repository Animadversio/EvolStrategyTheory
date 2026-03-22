#!/usr/bin/env python3
"""Exp A: Deep spectrum analysis of ZScoreES on quadratic landscapes."""
import sys, os, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA = 0.1; ALPHA = SIGMA / 2.0; N_POP = 50
OUT  = Path(os.path.expanduser("~/DL_Projects/EvolStrategyTheory_validation"))
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

def gc():
    if DEVICE.type == 'cuda':
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

def safe_chunk(d, N, gb=1.5):
    return max(20, int(gb * 1e9 / (4 * N * d)))

def make_rot(d, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R)); signs[signs==0] = 1.0
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

def measure_sigmaR(Rot, eigs, theta, theta_star, sigma_val, xi, N):
    d = eigs.shape[0]
    chunk = min(safe_chunk(d, 1, gb=2.0), 2000)
    dx = theta - theta_star
    y  = Rot @ dx
    v  = Rot.T @ (eigs * y)
    vnorm = v.norm().item()
    trQ2  = (eigs**2).sum().item()
    sR_th = np.sqrt(sigma_val**2*vnorm**2 + 0.5*sigma_val**4*trQ2 + xi**2)
    # empirical: sample chunk perturbations
    eps  = torch.randn(chunk, d, device=DEVICE)
    dx2  = theta.unsqueeze(0) + sigma_val * eps - theta_star.unsqueeze(0)
    y2   = dx2 @ Rot.T
    R    = -0.5 * (eigs * y2 * y2).sum(-1)
    if xi > 0: R = R + xi * torch.randn(chunk, device=DEVICE)
    sR_emp  = R.std().item()
    eff_step = ALPHA * sigma_val / (sR_th + 1e-30)
    del eps, dx2, y2, R; gc()
    return sR_th, sR_emp, eff_step, vnorm, trQ2

# ── A1: k/d ratio sweep ───────────────────────────────────────────────────────
print("=== A1: k/d sweep ===")
D_VALS = [100, 500, 1000]
KD_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.95]
spectra_labels = [('uniform','Uniform'), ('powerlaw','Power-law β=1'), ('range','Range')]

fig, axes = plt.subplots(1, len(D_VALS), figsize=(15, 5))
for ax_i, d in enumerate(D_VALS):
    Rot   = make_rot(d, seed=42)
    theta = 3.0 * torch.randn(d, device=DEVICE)
    ts    = torch.zeros(d, device=DEVICE)
    cmap  = plt.cm.tab10
    for sp_i, (sp_tag, sp_label) in enumerate(spectra_labels):
        sR_ths, sR_emps = [], []
        for frac in KD_FRACS:
            k = max(1, int(d * frac))
            eigs = make_eigs(d, k, sp_tag, lam_max=5.0, lam_min=0.5, flat=1e-4, beta=1.0)
            sR_th, sR_emp, _, _, _ = measure_sigmaR(Rot, eigs, theta, ts, SIGMA, 0.1, N_POP)
            sR_ths.append(sR_th); sR_emps.append(sR_emp)
            del eigs; gc()
        ks = [max(1, int(d*f)) for f in KD_FRACS]
        axes[ax_i].plot(ks, sR_ths, '--', color=cmap(sp_i/10), lw=2, label=f'{sp_label} th')
        axes[ax_i].plot(ks, sR_emps, 'o', color=cmap(sp_i/10), ms=7)
        print(f"  d={d} {sp_label}: sR range {min(sR_ths):.2f}—{max(sR_ths):.2f}")
    axes[ax_i].set_xlabel('k'); axes[ax_i].set_ylabel('σ_R')
    axes[ax_i].set_title(f'd={d}: σ_R vs k')
    axes[ax_i].legend(fontsize=7); axes[ax_i].grid(True, alpha=0.3)
    del Rot; gc()
plt.suptitle('Exp A1: σ_R vs k across spectra', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expA1_sigmaR_k_sweep.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expA1_sigmaR_k_sweep.png")

# ── A2: Power-law beta sweep ──────────────────────────────────────────────────
print("=== A2: Power-law beta sweep ===")
BETA_VALS = [0.25, 0.5, 1.0, 2.0, 4.0]
XI_VALS = [0.0, 0.1, 1.0]
d = 1000; k = 50
Rot   = make_rot(d, seed=7)
theta = 3.0 * torch.randn(d, device=DEVICE)
ts    = torch.zeros(d, device=DEVICE)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cmap2 = plt.cm.plasma
for xi_i, xi in enumerate(XI_VALS):
    sR_ths, eff_steps = [], []
    for beta in BETA_VALS:
        eigs = make_eigs(d, k, 'powerlaw', lam_max=5.0, flat=1e-4, beta=beta)
        sR_th, _, eff_step, _, _ = measure_sigmaR(Rot, eigs, theta, ts, SIGMA, xi, N_POP)
        sR_ths.append(sR_th); eff_steps.append(eff_step)
        del eigs; gc()
    c = cmap2(xi_i / max(1, len(XI_VALS)-1))
    axes[0].plot(BETA_VALS, sR_ths, 'o-', color=c, ms=8, label=f'ξ={xi}')
    axes[1].plot(BETA_VALS, eff_steps, 's-', color=c, ms=8, label=f'ξ={xi}')
    print(f"  xi={xi}: sR {[f'{s:.2f}' for s in sR_ths]}")

axes[0].set_xlabel('β'); axes[0].set_ylabel('σ_R')
axes[0].set_title(f'σ_R vs β (d={d}, k={k})'); axes[0].legend(fontsize=8); axes[0].grid(True,alpha=0.3)
axes[1].set_xlabel('β'); axes[1].set_ylabel('γ = ασ/σ_R')
axes[1].set_title(f'Effective step size vs β'); axes[1].legend(fontsize=8); axes[1].grid(True,alpha=0.3)
plt.suptitle(f'Exp A2: Power-law β sweep (d={d}, k={k})', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expA2_powerlaw_beta.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expA2_powerlaw_beta.png")
del Rot; gc()

# ── A3: 2D heatmap sigma_R(k, beta) ──────────────────────────────────────────
print("=== A3: 2D heatmap sigma_R(k, beta) ===")
d = 1000
K_RANGE = [5, 10, 20, 50, 100, 200, 400, 700]
Rot = make_rot(d, seed=3)
theta = 3.0 * torch.randn(d, device=DEVICE)
ts    = torch.zeros(d, device=DEVICE)

grid_th  = np.zeros((len(BETA_VALS), len(K_RANGE)))
grid_eff = np.zeros((len(BETA_VALS), len(K_RANGE)))
for bi, beta in enumerate(BETA_VALS):
    for ki, k in enumerate(K_RANGE):
        eigs = make_eigs(d, k, 'powerlaw', lam_max=5.0, flat=1e-4, beta=beta)
        sR_th, _, eff_step, _, _ = measure_sigmaR(Rot, eigs, theta, ts, SIGMA, 0.1, N_POP)
        grid_th[bi, ki] = sR_th; grid_eff[bi, ki] = eff_step
        del eigs; gc()
    print(f"  beta={beta}: sR = {[f'{grid_th[bi,ki]:.2f}' for ki in range(len(K_RANGE))]}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im0 = axes[0].imshow(grid_th, aspect='auto', origin='lower', cmap='hot')
axes[0].set_xticks(range(len(K_RANGE))); axes[0].set_xticklabels(K_RANGE)
axes[0].set_yticks(range(len(BETA_VALS))); axes[0].set_yticklabels(BETA_VALS)
axes[0].set_xlabel('k'); axes[0].set_ylabel('β')
axes[0].set_title(f'σ_R (theory) d={d} xi=0.1')
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(grid_eff, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_xticks(range(len(K_RANGE))); axes[1].set_xticklabels(K_RANGE)
axes[1].set_yticks(range(len(BETA_VALS))); axes[1].set_yticklabels(BETA_VALS)
axes[1].set_xlabel('k'); axes[1].set_ylabel('β')
axes[1].set_title(f'γ = ασ/σ_R d={d} xi=0.1')
plt.colorbar(im1, ax=axes[1])
plt.suptitle('Exp A3: 2D Heatmap — Spectrum Shape vs Gradient Efficiency', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expA3_heatmap_k_beta.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expA3_heatmap_k_beta.png")
del Rot; gc()

print("\nExp A complete!")
