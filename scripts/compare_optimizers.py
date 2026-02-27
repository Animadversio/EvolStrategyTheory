#!/usr/bin/env python3
"""
Compare Evolution Strategy and AdamW optimizers on rotated landscapes.

This script creates rotated optimization landscapes with k sensitive dimensions
and d-k flat dimensions, then compares ES and AdamW performance.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core.landscapes import RotatedQuadratic, RotatedGaussian
from core.optimizers import SeparableES
from core.utils import get_device, compute_metrics


def run_optimizer(optimizer, landscape, optimum, max_iters=1000,
                  eval_interval=1, optimizer_name="Optimizer"):
    """
    Run an optimizer and track metrics.

    Parameters
    ----------
    optimizer : optimizer instance
        Either ES or PyTorch optimizer (AdamW)
    landscape : callable
        Landscape function
    optimum : torch.Tensor
        Known optimum for tracking distance
    max_iters : int
        Maximum iterations
    eval_interval : int
        How often to record metrics
    optimizer_name : str
        Name for logging

    Returns
    -------
    dict
        History dictionary with tracked metrics
    """
    history = {
        'iterations': [],
        'loss': [],
        'distance': [],
        'grad_norm': [],
    }

    # Determine optimizer type
    is_es = hasattr(optimizer, 'ask')
    is_pytorch = hasattr(optimizer, 'param_groups')

    print(f"\n{'='*60}")
    print(f"Running {optimizer_name}")
    print(f"{'='*60}")

    for iteration in range(max_iters):
        if is_es:
            # Evolution Strategy
            info = optimizer.step(landscape)
            current_solution = optimizer.get_current_solution()

            if iteration % eval_interval == 0:
                metrics = compute_metrics(current_solution, landscape, optimum)
                history['iterations'].append(iteration)
                history['loss'].append(info['best_fitness'])
                history['distance'].append(metrics['distance'])
                history['grad_norm'].append(metrics['grad_norm'])

                if iteration % 50 == 0:
                    print(f"Iter {iteration:4d} | Loss: {info['best_fitness']:.6e} | "
                          f"Dist: {metrics['distance']:.6e} | Sigma: {info['sigma']:.4f}")

        elif is_pytorch:
            # PyTorch optimizer (AdamW)
            optimizer.zero_grad()

            # Get current parameters
            params = optimizer.param_groups[0]['params'][0]

            # Compute loss and gradient
            loss = landscape(params)
            loss.backward()

            # Step
            optimizer.step()

            if iteration % eval_interval == 0:
                metrics = compute_metrics(params, landscape, optimum)
                history['iterations'].append(iteration)
                history['loss'].append(metrics['loss'])
                history['distance'].append(metrics['distance'])
                history['grad_norm'].append(metrics['grad_norm'])

                if iteration % 50 == 0:
                    print(f"Iter {iteration:4d} | Loss: {metrics['loss']:.6e} | "
                          f"Dist: {metrics['distance']:.6e} | Grad: {metrics['grad_norm']:.6e}")

    print(f"\nFinal Loss: {history['loss'][-1]:.6e}")
    print(f"Final Distance: {history['distance'][-1]:.6e}")

    return history


def plot_comparison(histories, landscape_name, save_path=None):
    """
    Create comparison plots for multiple optimizers.

    Parameters
    ----------
    histories : dict
        Dictionary mapping optimizer names to history dictionaries
    landscape_name : str
        Name of the landscape for the title
    save_path : str or None
        Path to save figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Loss over iterations
    ax1 = fig.add_subplot(gs[0, 0])
    for name, history in histories.items():
        ax1.plot(history['iterations'], history['loss'], label=name, linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss Convergence - {landscape_name}', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance to optimum
    ax2 = fig.add_subplot(gs[0, 1])
    for name, history in histories.items():
        ax2.plot(history['iterations'], history['distance'], label=name, linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Distance to Optimum', fontsize=12)
    ax2.set_title(f'Distance to Optimum - {landscape_name}', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient norm
    ax3 = fig.add_subplot(gs[1, 0])
    for name, history in histories.items():
        ax3.plot(history['iterations'], history['grad_norm'], label=name, linewidth=2)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Gradient Norm', fontsize=12)
    ax3.set_title(f'Gradient Norm - {landscape_name}', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss comparison (last 20% of iterations)
    ax4 = fig.add_subplot(gs[1, 1])
    for name, history in histories.items():
        start_idx = int(len(history['iterations']) * 0.8)
        ax4.plot(history['iterations'][start_idx:],
                history['loss'][start_idx:],
                label=name, linewidth=2)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title(f'Loss (Final 20%) - {landscape_name}', fontsize=14)
    ax4.set_yscale('log')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare ES and AdamW optimizers')
    parser.add_argument('--dim', type=int, default=1000,
                       help='Total dimensionality (default: 1000)')
    parser.add_argument('--sensitive-dims', type=int, default=20,
                       help='Number of sensitive dimensions (default: 20)')
    parser.add_argument('--landscape', type=str, default='quadratic',
                       choices=['quadratic', 'gaussian'],
                       help='Landscape type (default: quadratic)')
    parser.add_argument('--max-iters', type=int, default=500,
                       help='Maximum iterations (default: 500)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu, cuda, mps, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--init-distance', type=float, default=10.0,
                       help='Initial distance from optimum (default: 10.0)')
    parser.add_argument('--es-population', type=int, default=None,
                       help='ES population size (default: auto)')
    parser.add_argument('--es-sigma', type=float, default=1.0,
                       help='ES initial sigma (default: 1.0)')
    parser.add_argument('--adamw-lr', type=float, default=0.01,
                       help='AdamW learning rate (default: 0.01)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save plot (default: None)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    # Create landscape
    print(f"\nCreating {args.landscape} landscape...")
    print(f"  Dimensions: {args.dim}")
    print(f"  Sensitive dimensions: {args.sensitive_dims}")

    if args.landscape == 'quadratic':
        landscape = RotatedQuadratic(
            dim=args.dim,
            sensitive_dims=args.sensitive_dims,
            eigenvalue_range=(1.0, 100.0),
            flat_eigenvalue=1e-6,
            device=device,
            seed=args.seed
        )
    else:  # gaussian
        landscape = RotatedGaussian(
            dim=args.dim,
            sensitive_dims=args.sensitive_dims,
            variance_range=(0.01, 1.0),
            flat_variance=100.0,
            device=device,
            seed=args.seed
        )

    optimum = landscape.optimum
    print(f"  Optimum location: norm = {torch.norm(optimum).item():.4f}")

    # Create initial point (same for both optimizers)
    init_point = optimum + args.init_distance * torch.randn(args.dim, device=device)
    init_point = init_point / torch.norm(init_point - optimum) * args.init_distance + optimum
    print(f"  Initial distance: {torch.norm(init_point - optimum).item():.4f}")

    # Initialize optimizers
    print("\nInitializing optimizers...")

    # ES optimizer
    es_optimizer = SeparableES(
        init_params=init_point.clone(),
        population_size=args.es_population,
        sigma=args.es_sigma,
        lr=1.0,
        sigma_lr=0.2,
        device=device
    )
    print(f"  ES population size: {es_optimizer.population_size}")

    # AdamW optimizer
    adamw_params = init_point.clone().requires_grad_(True)
    adamw_optimizer = torch.optim.AdamW([adamw_params], lr=args.adamw_lr)
    print(f"  AdamW learning rate: {args.adamw_lr}")

    # Run optimizers
    histories = {}

    histories['ES'] = run_optimizer(
        es_optimizer,
        landscape,
        optimum,
        max_iters=args.max_iters,
        eval_interval=1,
        optimizer_name="Evolution Strategy (ES)"
    )

    # Re-initialize AdamW params to same starting point
    adamw_params.data = init_point.clone()
    adamw_optimizer = torch.optim.AdamW([adamw_params], lr=args.adamw_lr)

    histories['AdamW'] = run_optimizer(
        adamw_optimizer,
        landscape,
        optimum,
        max_iters=args.max_iters,
        eval_interval=1,
        optimizer_name="AdamW"
    )

    # Plot comparison
    landscape_name = f"{args.landscape.capitalize()} (d={args.dim}, k={args.sensitive_dims})"
    plot_comparison(histories, landscape_name, save_path=args.save_plot)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, history in histories.items():
        final_loss = history['loss'][-1]
        final_dist = history['distance'][-1]
        print(f"\n{name}:")
        print(f"  Final Loss: {final_loss:.6e}")
        print(f"  Final Distance: {final_dist:.6e}")
        print(f"  Loss Reduction: {history['loss'][0] / final_loss:.2f}x")


if __name__ == '__main__':
    main()
