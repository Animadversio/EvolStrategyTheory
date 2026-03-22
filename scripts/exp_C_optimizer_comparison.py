#!/usr/bin/env python3
"""Exp C: Compare ZScoreES, SeparableES (CMA-style), and SimpleES on rotated quadratic landscapes.

Benchmarks three ES variants across conditioning, dimensionality, and
population size to understand where ZScoreES's z-score normalization helps.

Sub-experiments:
  C1: Well-conditioned landscape (κ=10), d=500, k=20, T=300, ξ=0.1
  C2: Ill-conditioned landscape (κ=1000), d=500, k=20, T=300, ξ=0.1
      → SimpleES diverges catastrophically; ZScoreES is robust.
  C3: Varying dimensionality d ∈ {100,500,1000}, κ=100, k=20, T=200, ξ=0.1
      → Final distance ∝ √d; ZScoreES edge widens at high d.
  C4: Varying population N ∈ {10,30,100,300} for ZScoreES, d=500, κ=100, T=200, ξ=0.1
      → Diminishing returns beyond N ≈ 2k; N=30 sufficient here.

Requires: core/optimizers.py (ZScoreES, SeparableES, SimpleES)
          core/landscapes.py (RotatedQuadratic)

Hardcoded settings: see conditions above.

Outputs:
  ~/DL_Projects/EvolStrategyTheory_validation/figures/expC{12,3,4}_*.png

Usage:
  python scripts/exp_C_optimizer_comparison.py
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
from core.optimizers import SeparableES, ZScoreES, SimpleES
from core.landscapes import RotatedQuadratic

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT  = Path(os.path.expanduser("~/DL_Projects/EvolStrategyTheory_validation"))
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

def gc():
    if DEVICE.type == 'cuda':
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

XI = 0.1

def run_optimizer(opt_class, opt_kwargs, landscape, xi, T, device):
    """Run optimizer for T steps, return distance trajectory."""
    opt = opt_class(**opt_kwargs)
    dist_traj = []
    for t in range(T):
        if opt_class == ZScoreES:
            candidates, perturbations = opt.ask()
            rewards = torch.tensor(
                [landscape(c).item() for c in candidates], device=device)
            # add obs noise
            rewards = rewards + xi * torch.randn_like(rewards)
            opt.tell(perturbations, rewards)
        elif opt_class == SeparableES:
            population, perturbations = opt.ask()
            fitness = torch.tensor(
                [landscape(c).item() for c in population], device=device)
            fitness = fitness + xi * torch.randn_like(fitness)
            opt.tell(perturbations, fitness)
        else:  # SimpleES
            def noisy_landscape(x):
                return landscape(x) + xi * torch.randn(1, device=device).item()
            opt.step(noisy_landscape)
        soln = opt.get_current_solution()
        dist_traj.append((soln - landscape.optimum).norm().item())
    return dist_traj

# ── C1 & C2: well vs ill conditioned ─────────────────────────────────────────
print("=== C1/C2: Well vs Ill conditioned ===")
D, K, T = 500, 20, 300
N_POP = 50
CONFIGS = [
    ('well-cond κ=10',   dict(dim=D, sensitive_dims=K, eigenvalue_range=(1.0, 10.0),  flat_eigenvalue=1e-4, device=DEVICE, seed=42)),
    ('ill-cond κ=1000',  dict(dim=D, sensitive_dims=K, eigenvalue_range=(1.0, 1000.0), flat_eigenvalue=1e-4, device=DEVICE, seed=42)),
    ('powerlaw approx',  dict(dim=D, sensitive_dims=K, eigenvalue_range=(1.0, 100.0),  flat_eigenvalue=1e-4, device=DEVICE, seed=42)),
]
optimizers = [
    ('ZScoreES',    ZScoreES,    lambda d,dev: dict(init_params=5.0*torch.randn(d,device=dev), population_size=N_POP, sigma=0.5, device=dev, reward_mode='minimize', antithetic=True)),
    ('SeparableES', SeparableES, lambda d,dev: dict(init_params=5.0*torch.randn(d,device=dev), population_size=N_POP, sigma=0.5, device=dev)),
    ('SimpleES',    SimpleES,    lambda d,dev: dict(init_params=5.0*torch.randn(d,device=dev), population_size=N_POP, sigma=0.1, lr=0.01, device=dev)),
]

fig, axes = plt.subplots(1, len(CONFIGS), figsize=(18, 5))
torch.manual_seed(0)
for ci, (cfg_label, cfg_kwargs) in enumerate(CONFIGS):
    landscape = RotatedQuadratic(**cfg_kwargs)
    print(f"\n  {cfg_label}:")
    for opt_label, opt_class, opt_kwargs_fn in optimizers:
        torch.manual_seed(1)
        try:
            kwargs = opt_kwargs_fn(D, DEVICE)
            traj = run_optimizer(opt_class, kwargs, landscape, XI, T, DEVICE)
            axes[ci].semilogy(range(T), traj, lw=2, label=opt_label)
            print(f"    {opt_label}: final dist={traj[-1]:.4f}")
            del kwargs; gc()
        except Exception as e:
            print(f"    {opt_label}: ERROR {e}")
    axes[ci].set_xlabel('Step'); axes[ci].set_ylabel('||θ - θ*||')
    axes[ci].set_title(f'{cfg_label}\nd={D}, k={K}, xi={XI}')
    axes[ci].legend(fontsize=8); axes[ci].grid(True, alpha=0.3)
    del landscape; gc()

plt.suptitle('Exp C1/C2: Optimizer comparison on quadratic landscapes', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expC12_optimizer_comparison.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expC12_optimizer_comparison.png")

# ── C3: Varying dimensionality ────────────────────────────────────────────────
print("\n=== C3: Varying d ===")
D_VALS = [100, 500, 1000]
K = 20; T = 200; N_POP = 50

fig, axes = plt.subplots(1, len(D_VALS), figsize=(18, 5))
for di, d in enumerate(D_VALS):
    landscape = RotatedQuadratic(dim=d, sensitive_dims=K, eigenvalue_range=(1.0, 100.0),
                                  flat_eigenvalue=1e-4, device=DEVICE, seed=42)
    for opt_label, opt_class, opt_kwargs_fn in optimizers:
        torch.manual_seed(1)
        try:
            kwargs = opt_kwargs_fn(d, DEVICE)
            traj = run_optimizer(opt_class, kwargs, landscape, XI, T, DEVICE)
            axes[di].semilogy(range(T), traj, lw=2, label=opt_label)
            print(f"  d={d} {opt_label}: final dist={traj[-1]:.4f}")
            del kwargs; gc()
        except Exception as e:
            print(f"  d={d} {opt_label}: ERROR {e}")
    axes[di].set_xlabel('Step'); axes[di].set_ylabel('||θ - θ*||')
    axes[di].set_title(f'd={d}, k={K}, κ=100, xi={XI}')
    axes[di].legend(fontsize=8); axes[di].grid(True, alpha=0.3)
    del landscape; gc()

plt.suptitle('Exp C3: Optimizer comparison vs dimensionality', fontsize=13)
plt.tight_layout(); fig.savefig(FIGS/'expC3_vs_dim.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
print("  → saved expC3_vs_dim.png")

# ── C4: Varying N ────────────────────────────────────────────────────────────
print("\n=== C4: Varying N (ZScoreES only) ===")
N_VALS = [10, 30, 100, 300]
D, K, T = 500, 20, 200
landscape = RotatedQuadratic(dim=D, sensitive_dims=K, eigenvalue_range=(1.0,100.0),
                              flat_eigenvalue=1e-4, device=DEVICE, seed=42)
fig, ax = plt.subplots(figsize=(8, 5))
cmap4 = plt.cm.viridis
for ni, N in enumerate(N_VALS):
    torch.manual_seed(1)
    kwargs = dict(init_params=5.0*torch.randn(D,device=DEVICE), population_size=N,
                  sigma=0.5, device=DEVICE, reward_mode='minimize', antithetic=(N%2==0))
    traj = run_optimizer(ZScoreES, kwargs, landscape, XI, T, DEVICE)
    ax.semilogy(range(T), traj, color=cmap4(ni/max(1,len(N_VALS)-1)), lw=2, label=f'N={N}')
    print(f"  N={N}: final dist={traj[-1]:.4f}")
    del kwargs; gc()

ax.set_xlabel('Step'); ax.set_ylabel('||θ - θ*||')
ax.set_title(f'ZScoreES: Effect of N\n(d={D}, k={K}, κ=100, σ=0.5, xi={XI})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(FIGS/'expC4_N_scaling.png', dpi=130, bbox_inches='tight'); plt.close(fig); gc()
del landscape; gc()
print("  → saved expC4_N_scaling.png")

print("\nExp C complete!")
