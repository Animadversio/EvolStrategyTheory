"""
Simple demo script for ES vs AdamW comparison.
No CLI arguments - just modify the parameters directly in the script.
Results are available in the global scope for interactive analysis.

This version additionally tracks and plots the angle (cosine similarity)
between the update direction from the initial state to each subsequent state,
for both ES and AdamW.
"""
#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core.landscapes import RotatedQuadratic, RotatedGaussian
from core.optimizers import SeparableES
from core.utils import get_device, compute_metrics

# ============================================================================
# CONFIGURATION - Modify these parameters
# ============================================================================
#%%
# Landscape settings
DIM = 1000                    # Total dimensionality
SENSITIVE_DIMS = 5            # Number of sensitive dimensions (k << d)
LANDSCAPE_TYPE = 'quadratic'  # 'quadratic' or 'gaussian'
SEED = 42                     # Random seed

# Optimization settings
MAX_ITERS = 500               # Maximum iterations
INIT_DISTANCE = 10.0          # Initial distance from optimum

# ES settings
ES_POPULATION = None          # Population size (None = auto)
ES_SIGMA = 0.1                # Initial step size
ES_LR = 1.0                   # Learning rate for mean updates
ES_SIGMA_LR = 0.2             # Learning rate for sigma adaptation

# AdamW settings
ADAMW_LR = 0.01               # Learning rate

# Device
DEVICE = 'cpu'#None                 # None = auto, or 'cpu', 'cuda', 'mps'

# Visualization
EVAL_INTERVAL = 1             # How often to record metrics
SHOW_PLOTS = True             # Whether to display plots
SAVE_PLOT = None              # Path to save plot (None = don't save)

# ============================================================================
# SETUP
# ============================================================================

print("=" * 70)
print("ES vs AdamW Comparison on Rotated Landscapes")
print("=" * 70)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Get device
device = get_device(DEVICE)
print(f"\nDevice: {device}")

# Create landscape
print(f"\nCreating {LANDSCAPE_TYPE} landscape...")
print(f"  Dimensions: {DIM}")
print(f"  Sensitive dimensions: {SENSITIVE_DIMS}")
print(f"  Flat dimensions: {DIM - SENSITIVE_DIMS}")

if LANDSCAPE_TYPE == 'quadratic':
    landscape = RotatedQuadratic(
        dim=DIM,
        sensitive_dims=SENSITIVE_DIMS,
        eigenvalue_range=(1.0, 100.0),
        flat_eigenvalue=1e-6,
        device=device,
        seed=SEED
    )
    print(f"  Eigenvalue range (sensitive): [1.0, 100.0]")
    print(f"  Eigenvalue (flat): 1e-6")
    print(f"  Condition number: ~{100.0 / 1e-6:.2e}")
else:
    landscape = RotatedGaussian(
        dim=DIM,
        sensitive_dims=SENSITIVE_DIMS,
        variance_range=(0.01, 1.0),
        flat_variance=100.0,
        device=device,
        seed=SEED
    )
    print(f"  Variance range (sensitive): [0.01, 1.0]")
    print(f"  Variance (flat): 100.0")

optimum = landscape.optimum
print(f"  Optimum norm: {torch.norm(optimum).item():.4f}")

# Create initial point (same for both optimizers)
init_point = optimum + INIT_DISTANCE * torch.randn(DIM, device=device)
init_point = init_point / torch.norm(init_point - optimum) * INIT_DISTANCE + optimum
init_distance = torch.norm(init_point - optimum).item()
print(f"\nInitial distance from optimum: {init_distance:.4f}")
print(f"Initial loss: {landscape(init_point).item():.6e}")

# ============================================================================
# EVOLUTION STRATEGY
# ============================================================================

print("\n" + "=" * 70)
print("Running Evolution Strategy")
print("=" * 70)

# Store initial position for delta calculations
es_init_point = init_point.clone().detach()

# Initialize ES
es_optimizer = SeparableES(
    init_params=es_init_point.clone(),
    population_size=ES_POPULATION,
    sigma=ES_SIGMA,
    lr=ES_LR,
    sigma_lr=ES_SIGMA_LR,
    device=device
)
print(f"Population size: {es_optimizer.population_size}")
print(f"Initial sigma: {ES_SIGMA}")

# Run ES and record position history for angle calculation
es_history = {
    'iterations': [],
    'loss': [],
    'distance': [],
    'grad_norm': [],
    'sigma': [],
    'angle_deg': [],   # angle between delta vectors (degrees)
    'positions': [],
}

# For ES: need initial delta (init_point -> first step solution)
es_positions = []  # store positions at EVAL_INTERVAL for angle plot

for iteration in range(MAX_ITERS):
    info = es_optimizer.step(landscape)
    current_solution = es_optimizer.get_current_solution()
    
    if iteration % EVAL_INTERVAL == 0:
        metrics = compute_metrics(current_solution, landscape, optimum)
        es_history['iterations'].append(iteration)
        es_history['loss'].append(info['best_fitness'])
        es_history['distance'].append(metrics['distance'])
        es_history['grad_norm'].append(metrics['grad_norm'])
        es_history['sigma'].append(info['sigma'])
        es_history['positions'].append(current_solution.clone().detach())
        es_positions.append(current_solution.clone().detach())

        if iteration % 50 == 0:
            print(f"Iter {iteration:4d} | Loss: {info['best_fitness']:.6e} | "
                  f"Dist: {metrics['distance']:.6e} | Sigma: {info['sigma']:.4f}")


es_final_solution = es_optimizer.get_current_solution()
es_final_loss = es_history['loss'][-1]
es_final_distance = es_history['distance'][-1]

print(f"\nES Final Results:")
print(f"  Loss: {es_final_loss:.6e}")
print(f"  Distance: {es_final_distance:.6e}")
print(f"  Loss reduction: {es_history['loss'][0] / es_final_loss:.2f}x")

# ============================================================================
# ADAMW
# ============================================================================

print("\n" + "=" * 70)
print("Running AdamW")
print("=" * 70)

# Store initial for AdamW delta
adamw_init_point = init_point.clone().detach()
adamw_params = adamw_init_point.clone().requires_grad_(True)
adamw_optimizer = torch.optim.AdamW([adamw_params], lr=ADAMW_LR)
print(f"Learning rate: {ADAMW_LR}")

adamw_history = {
    'iterations': [],
    'loss': [],
    'distance': [],
    'grad_norm': [],
    'angle_deg': [],  # angle trajectory
}

# For AdamW: store positions for delta calculations
adamw_positions = []  # at each eval step

for iteration in range(MAX_ITERS):
    adamw_optimizer.zero_grad()

    loss = landscape(adamw_params)
    loss.backward()
    adamw_optimizer.step()

    if iteration % EVAL_INTERVAL == 0:
        metrics = compute_metrics(adamw_params, landscape, optimum)
        adamw_history['iterations'].append(iteration)
        adamw_history['loss'].append(metrics['loss'])
        adamw_history['distance'].append(metrics['distance'])
        adamw_history['grad_norm'].append(metrics['grad_norm'])
        adamw_positions.append(adamw_params.detach().clone())

        if iteration % 50 == 0:
            print(f"Iter {iteration:4d} | Loss: {metrics['loss']:.6e} | "
                  f"Dist: {metrics['distance']:.6e} | Grad: {metrics['grad_norm']:.6e}")


adamw_final_solution = adamw_params.detach().clone()
adamw_final_loss = adamw_history['loss'][-1]
adamw_final_distance = adamw_history['distance'][-1]

print(f"\nAdamW Final Results:")
print(f"  Loss: {adamw_final_loss:.6e}")
print(f"  Distance: {adamw_final_distance:.6e}")
print(f"  Loss reduction: {adamw_history['loss'][0] / adamw_final_loss:.2f}x")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"\nFinal Loss:")
print(f"  ES:    {es_final_loss:.6e}")
print(f"  AdamW: {adamw_final_loss:.6e}")
print(f"  Winner: {'ES' if es_final_loss < adamw_final_loss else 'AdamW'} "
      f"({min(es_final_loss, adamw_final_loss) / max(es_final_loss, adamw_final_loss):.2%} of other)")

print(f"\nFinal Distance to Optimum:")
print(f"  ES:    {es_final_distance:.6e}")
print(f"  AdamW: {adamw_final_distance:.6e}")
print(f"  Winner: {'ES' if es_final_distance < adamw_final_distance else 'AdamW'}")

print(f"\nConvergence Speed (iterations to 10x loss reduction):")
es_10x_iter = next((i for i, l in enumerate(es_history['loss'])
                   if l < es_history['loss'][0] / 10), len(es_history['loss']))
adamw_10x_iter = next((i for i, l in enumerate(adamw_history['loss'])
                      if l < adamw_history['loss'][0] / 10), len(adamw_history['loss']))
print(f"  ES:    {es_10x_iter} iterations")
print(f"  AdamW: {adamw_10x_iter} iterations")


# ============================================================================
# VISUALIZATION
# ============================================================================

# Compute ES projection statistics onto initial AdamW delta
# Only if we have es_positions and adamw_positions
es_proj_cosines = []
es_proj_norms = []
try:
    # es_positions and adamw_positions are lists of parameter tensors at every eval step
    if len(adamw_positions) >= 2 and 'positions' in es_history and len(es_history['positions']) >= 2:
        adamw_delta0 = adamw_positions[-1] - adamw_positions[0]
        adamw_delta0_norm = torch.norm(adamw_delta0)
        for es_pos in es_history['positions']:
            es_delta = es_pos - adamw_positions[0]  # project from same starting point
            if torch.norm(es_delta) < 1e-12 or adamw_delta0_norm < 1e-12:
                es_proj_cosines.append(0.0)
                es_proj_norms.append(0.0)
            else:
                # Cosine similarity
                cos_sim = torch.clamp(torch.dot(es_delta, adamw_delta0) /
                                     (torch.norm(es_delta) * adamw_delta0_norm), -1.0, 1.0)
                es_proj_cosines.append(cos_sim.item())
                # Norm of projection onto adamw_delta0 direction
                proj_length = torch.dot(es_delta, adamw_delta0) / adamw_delta0_norm
                es_proj_norms.append(proj_length.item())
    else:
        es_proj_cosines = [0.0 for _ in es_history['iterations']]
        es_proj_norms = [0.0 for _ in es_history['iterations']]
except Exception as e:
    print("Could not compute ES projections onto AdamW delta (missing or incompatible data):", e)
    es_proj_cosines = [0.0 for _ in es_history['iterations']]
    es_proj_norms = [0.0 for _ in es_history['iterations']]


print(f"\nFinal ES projection onto AdamW delta:")
# Print final parameter delta norm for ES and AdamW
es_param_delta_norm = torch.norm(es_history['positions'][-1] - es_history['positions'][0]) if 'positions' in es_history and len(es_history['positions']) >= 2 else float('nan')
adamw_param_delta_norm = torch.norm(adamw_positions[-1] - adamw_positions[0]) if len(adamw_positions) >= 2 else float('nan')
print(f"Final Parameter Delta Norm:")
print(f"  ES:    {es_param_delta_norm:.6e}")
print(f"  AdamW: {adamw_param_delta_norm:.6e}")
print(f"  Final Cosine:   {es_proj_cosines[-1]:.6f}")
print(f"  Final ProjNorm: {es_proj_norms[-1]:.6e}")


if SHOW_PLOTS:
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.33)

    # Plot 1: Loss over iterations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(es_history['iterations'], es_history['loss'],
            label='ES', linewidth=2, marker='o', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax1.plot(adamw_history['iterations'], adamw_history['loss'],
            label='AdamW', linewidth=2, marker='s', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance to optimum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(es_history['iterations'], es_history['distance'],
            label='ES', linewidth=2, marker='o', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax2.plot(adamw_history['iterations'], adamw_history['distance'],
            label='AdamW', linewidth=2, marker='s', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Distance to Optimum', fontsize=12)
    ax2.set_title('Distance to Optimum', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient norm
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(es_history['iterations'], es_history['grad_norm'],
            label='ES', linewidth=2, marker='o', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax3.plot(adamw_history['iterations'], adamw_history['grad_norm'],
            label='AdamW', linewidth=2, marker='s', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Gradient Norm', fontsize=12)
    ax3.set_title('Gradient Norm', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: ES Sigma adaptation
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(es_history['iterations'], es_history['sigma'],
            linewidth=2, color='green', marker='o', markersize=3, markevery=max(1, MAX_ITERS//20))
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Sigma', fontsize=12)
    ax4.set_title('ES Step Size Adaptation', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Loss (final 20%)
    ax5 = fig.add_subplot(gs[1, 1])
    start_idx = int(len(es_history['iterations']) * 0.8)
    ax5.plot(es_history['iterations'][start_idx:], es_history['loss'][start_idx:],
            label='ES', linewidth=2, marker='o', markersize=3, markevery=max(1, (MAX_ITERS-start_idx)//10))
    ax5.plot(adamw_history['iterations'][start_idx:], adamw_history['loss'][start_idx:],
            label='AdamW', linewidth=2, marker='s', markersize=3, markevery=max(1, (MAX_ITERS-start_idx)//10))
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Loss (Final 20%)', fontsize=14, fontweight='bold')
    ax5.set_yscale('log')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Optimization trajectory in sensitive subspace (2D projection)
    ax6 = fig.add_subplot(gs[1, 2])
    es_trajectory = []
    adamw_trajectory = []
    # Sample trajectory points
    sample_indices = np.linspace(0, len(es_history['iterations'])-1, 20, dtype=int)
    # We'll need to store trajectories - let's do a simple visualization instead
    ax6.text(0.5, 0.5, 'Trajectory visualization\nrequires storing\nintermediate solutions',
            ha='center', va='center', fontsize=12, transform=ax6.transAxes)
    ax6.set_title('Trajectory (Placeholder)', fontsize=14, fontweight='bold')
    ax6.axis('off')

    # Plot 8: ES projection statistics onto AdamW initial delta
    ax8 = fig.add_subplot(gs[2, :])
    ax8.plot(es_history['iterations'], es_proj_cosines, label='ES-to-AdamW Initial Δ Cosine', color='C2', linewidth=2)
    ax8.plot(es_history['iterations'], es_proj_norms, label='ES proj. norm on AdamW Initial Δ', color='C3', linewidth=2)
    ax8.set_xlabel("Iteration", fontsize=12)
    ax8.set_ylabel("Value", fontsize=12)
    ax8.set_title("ES Projection onto AdamW Initial Δ-direction", fontsize=14, fontweight='bold')
    ax8.legend(fontsize=12)
    ax8.grid(True, alpha=0.3)

    plt.suptitle(f'{LANDSCAPE_TYPE.capitalize()} Landscape (d={DIM}, k={SENSITIVE_DIMS})',
                fontsize=16, fontweight='bold', y=0.995)

    if SAVE_PLOT:
        plt.savefig(SAVE_PLOT, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {SAVE_PLOT}")

    plt.show()

print("\n" + "=" * 70)
print("All results are available in the global scope:")
print("  - landscape: the optimization landscape")
print("  - optimum: the true optimum location")
print("  - es_history: ES optimization history")
print("  - adamw_history: AdamW optimization history")
print("  - es_final_solution: ES final solution")
print("  - adamw_final_solution: AdamW final solution")
print("=" * 70)

# %%
