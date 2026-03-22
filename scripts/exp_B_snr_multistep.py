#!/usr/bin/env python3
"""Exp B: SNR scaling for ρ (Prop 2 deep dive) + multi-step trajectory analysis.

Deep validation of Proposition 2 (on-manifold fraction ρ) and multi-step
convergence on quadratic landscapes from theory.tex.

Sub-experiments:
  B1: ρ heatmap over (d, ξ) — theory vs empirical across
      d ∈ {50,100,500,1000,5000} and ξ ∈ {0,0.01,...,10}.
  B2: Multi-step trajectory analysis — ‖θ_t - θ*‖, σ_R(t), ‖E[Δθ]‖
      for varying obs noise ξ on a d=500 quadratic landscape.
  B3: Required population N* for target ρ — analytic scaling N* ≈ ρd/(s(1-ρ)),
      showing N* grows linearly with d.

Key finding: ρ ≈ N·s/d; curse of dimensionality is severe; N* ∝ d.

Hardcoded settings: σ=0.1, α=σ/2, N=50 (for B1/B2).

Outputs:
  ~/DL_Projects/EvolStrategyTheory_validation/figures/expB{1-3}_*.png

Usage:
  python scripts/exp_B_snr_multistep.py
  # (no CLI args — edit constants at top of file to change settings)
"""
import sys, os, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA = 0.1; ALPHA = SIGMA / 2.0
OUT  = Path(os.path.expanduser("~/DL_Projects/EvolStrategyTheory_validation"))
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

def gc():
    if DEVICE.type == 'cuda':
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

def th_rho(s, d, N): return (1 + (N+1)*s) / (d + (N+1)*s)
def snr(sigma, vnorm, xi): return sigma**2*vnorm**2 / (sigma**2*vnorm**2 + xi**2 + 1e-30)

def make_rot(d, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R)); signs[signs==0] = 1.0
    return (Q * signs.unsqueeze(0)).to(DEVICE)

def make_eigs(d, k, lam_max=5.0, flat=1e-4, beta=1.0, spectrum='uniform'):
    eigs = torch.full((d,), flat, dtype=torch.float32)
    if spectrum == 'uniform': eigs[:k] = lam_max
    elif spectrum == 'powerlaw':
        eigs[:k] = torch.tensor([lam_max*(i+1)**(-beta) for i in range(k)], dtype=torch.float32)
    return eigs.to(DEVICE)

# ── B1: rho vs SNR heatmap (d vs xi) ─────────────────────────────────────────
print("=== B1: rho vs d and xi (SNR heatmap) ===")
D_VALS = [50, 100, 500, 1000, 5000]
XI_VALS = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
N_FIXED = 50
VNORM = 1.0

rho_th_grid  = np.zeros((len(D_VALS), len(XI_VALS)))
rho_emp_grid = np.zeros((len(D_VALS), len(XI_VALS)))

for di, d in enumerate(D_VALS):
    v     = torch.zeros(d, device=DEVICE); v[0] = VNORM
    v_hat = v / v.norm()
    chunk = max(50, int(1.5e9 / (4 * N_FIXED * d)))
    for xi_i, xi in enumerate(XI_VALS):
        s = snr(SIGMA, VNORM, xi)
        rho_th_grid[di, xi_i] = th_rho(s, d, N_FIXED)
        # empirical
        nmc = min(chunk, 2000)
        eps = torch.randn(nmc, N_FIXED, d, device=DEVICE)
        R   = -SIGMA * (eps @ v)
        if xi > 0: R = R + xi * torch.randn(nmc, N_FIXED, device=DEVICE)
        Z   = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-9)
        dth = (ALPHA / N_FIXED) * (Z.unsqueeze(-1) * eps).sum(1)  # (nmc, d)
        proj = dth @ v_hat
        par  = proj.unsqueeze(-1) * v_hat.unsqueeze(0)
        perp = dth - par
        rho_emp_grid[di, xi_i] = (par**2).sum(-1).mean().item() / ((dth**2).sum(-1).mean().item() + 1e-30)
        del eps, R, Z, dth, par, perp; gc()
    print(f"  d={d}: rho_th={[f'{rho_th_grid[di,xi_i]:.4f}' for xi_i in range(len(XI_VALS))]}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
im0 = axes[0].imshow(np.log10(rho_th_grid + 1e-10), aspect='auto', origin='lower', cmap='viridis')
axes[0].set_xticks(range(len(XI_VALS))); axes[0].set_xticklabels(XI_VALS)
axes[0].set_yticks(range(len(D_VALS))); axes[0].set_yticklabels(D_VALS)
axes[0].set_xlabel('ξ (obs noise)'); axes[0].set_ylabel('d')
axes[0].set_title('log10(ρ) theory'); plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(np.log10(rho_emp_grid + 1e-10), aspect='auto', origin='lower', cmap='viridis')
axes[1].set_xticks(range(len(XI_VALS))); axes[1].set_xticklabels(XI_VALS)
axes[1].set_yticks(range(len(D_VALS))); axes[1].set_yticklabels(D_VALS)
axes[1].set_xlabel('ξ'); axes[1].set_ylabel('d')
axes[1].set_title('log10(ρ) empirical'); plt.colorbar(im1, ax=axes[1])

# ratio emp/theory
ratio = rho_emp_grid / (rho_th_grid + 1e-10)
im2 = axes[2].imshow(ratio, aspect='auto', origin='lower', cmap='RdBu', vmin=0.8, vmax=1.2)
axes[2].set_xticks(range(len(XI_VALS))); axes[2].set_xticklabels(XI_VALS)
axes[2].set_yticks(range(len(D_VALS))); axes[2].set_yticklabels(D_VALS)
axes[2].set_xlabel('ξ'); axes[2].set_ylabel('d')
axes[2].set_title('ρ emp/theory ratio'); plt.colorbar(im2, ax=axes[2])

plt.suptitle('Exp B1: ρ heatmap (d × ξ), N=50', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expB1_rho_heatmap.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expB1_rho_heatmap.png")

# ── B2: Multi-step: empirical theta trajectory vs theoretical prediction ──────
# Compare mean update magnitude vs theory over steps on quadratic landscape
print("=== B2: Multi-step mean update tracking ===")
d = 500; k = 20; N = 50; T = 200
XI_MULTI = [0.0, 0.1, 1.0, 5.0]

Rot  = make_rot(d, seed=42)
eigs = make_eigs(d, k, lam_max=5.0, flat=1e-4, spectrum='uniform')
ts   = torch.zeros(d, device=DEVICE)

trQ2 = (eigs**2).sum().item()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmap_xi = plt.cm.coolwarm

for xi_i, xi in enumerate(XI_MULTI):
    theta = 5.0 * torch.ones(d, device=DEVICE)
    dist_traj, sR_traj, sR_th_traj, mean_up_traj = [], [], [], []
    for t in range(T):
        # analytic predictions at current theta
        dx   = theta - ts
        y    = Rot @ dx
        v    = Rot.T @ (eigs * y)
        vnorm = v.norm().item()
        sR_th_cur = np.sqrt(SIGMA**2*vnorm**2 + 0.5*SIGMA**4*trQ2 + xi**2)
        mean_up_th = ALPHA * SIGMA * vnorm / (sR_th_cur + 1e-30)
        # empirical step
        n_samp = min(max(50, int(1e9/(4*d))), 2000)
        eps  = torch.randn(n_samp, d, device=DEVICE)
        dx2  = theta.unsqueeze(0) + SIGMA * eps - ts.unsqueeze(0)
        y2   = dx2 @ Rot.T
        R    = -0.5 * (eigs * y2 * y2).sum(-1)
        if xi > 0: R = R + xi * torch.randn(n_samp, device=DEVICE)
        sR_emp = R.std().item()
        Z    = (R - R.mean()) / (sR_emp + 1e-9)
        dth  = (ALPHA / n_samp) * (Z.unsqueeze(-1) * eps).sum(0)
        theta = theta + dth
        dist_traj.append((theta - ts).norm().item())
        sR_traj.append(sR_emp)
        sR_th_traj.append(sR_th_cur)
        mean_up_traj.append(mean_up_th)
        del eps, dx2, y2, R, Z, dth; gc()
    c = cmap_xi(xi_i / max(1, len(XI_MULTI)-1))
    axes[0].semilogy(range(T), dist_traj, color=c, lw=2, label=f'ξ={xi}')
    axes[1].plot(range(T), sR_traj, color=c, lw=1.5, label=f'ξ={xi} emp')
    axes[1].plot(range(T), sR_th_traj, '--', color=c, lw=1.5, alpha=0.7, label=f'ξ={xi} th')
    axes[2].plot(range(T), mean_up_traj, color=c, lw=2, label=f'ξ={xi}')
    print(f"  xi={xi}: final dist={dist_traj[-1]:.3f}  initial sR_th={sR_th_traj[0]:.3f}  final sR_th={sR_th_traj[-1]:.3f}")

axes[0].set_xlabel('Step t'); axes[0].set_ylabel('||θ_t - θ*||')
axes[0].set_title(f'Multi-step convergence\n(d={d}, k={k}, N={N})'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Step t'); axes[1].set_ylabel('σ_R')
axes[1].set_title('σ_R (emp+theory) over steps\n(shows dynamic attenuation)'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
axes[2].set_xlabel('Step t'); axes[2].set_ylabel('||E[Δθ]|| (theory)')
axes[2].set_title('Theoretical mean update magnitude\n(decreases as θ→θ*)'); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Exp B2: Multi-step Trajectory Analysis (d={d}, k={k})', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expB2_multistep_trajectory.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expB2_multistep_trajectory.png")
del Rot, eigs; gc()

# ── B3: N scaling — how many samples needed for rho to matter? ───────────────
print("=== B3: N scaling — break-even ===")
# Find N* such that rho = 0.5 (half movement on-manifold)
# rho=0.5 => 1 + (N+1)s = 0.5*(d + (N+1)s) => N+1 = (d-2)/(2s-1) ~ d/s for large d
D_RANGE = [100, 500, 1000, 5000, 10000]
S_VALS  = [0.01, 0.1, 0.5, 1.0]

fig, ax = plt.subplots(figsize=(8, 5))
cmap3 = plt.cm.tab10
for si, s in enumerate(S_VALS):
    # N needed for rho > target
    targets = [0.1, 0.5]
    for ti, target_rho in enumerate(targets):
        N_stars = []
        for d in D_RANGE:
            # solve (1+(N+1)*s)/(d+(N+1)*s) = target => N+1 = (target*d - 1)/(s*(1-target))
            if s * (1 - target_rho) < 1e-10:
                N_stars.append(np.inf)
            else:
                N_star = (target_rho * d - 1) / (s * (1 - target_rho)) - 1
                N_stars.append(max(0, N_star))
        ls = '-' if ti == 0 else '--'
        ax.loglog(D_RANGE, N_stars, ls, color=cmap3(si/10), lw=2,
                  label=f's={s}, ρ>{target_rho}')
        print(f"  s={s} ρ>{target_rho}: N* = {[f'{n:.0f}' for n in N_stars]}")

ax.axhline(50, color='k', ls=':', lw=2, label='N=50 (typical)')
ax.axhline(1000, color='gray', ls=':', lw=1.5, label='N=1000')
ax.set_xlabel('d'); ax.set_ylabel('N* (required population size)')
ax.set_title('Required N to achieve target ρ\n(N* ≈ d/(s*(1-ρ)) — grows linearly with d!)')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'expB3_N_scaling.png', dpi=130, bbox_inches='tight'); plt.close(fig)
print("  → saved expB3_N_scaling.png")

print("\nExp B complete!")
