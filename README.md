# Evolution Strategy Theory

Research codebase for analyzing evolutionary strategy theory, with focus on order statistics of Gaussian distributions and optimization landscape analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Compare ES and AdamW on Rotated Landscapes

```bash
# Run with default settings (1000-dim quadratic, 20 sensitive dims)
python scripts/compare_optimizers.py

# Customize settings
python scripts/compare_optimizers.py --dim 500 --sensitive-dims 10 --max-iters 1000

# Use Gaussian landscape
python scripts/compare_optimizers.py --landscape gaussian

# Save plots
python scripts/compare_optimizers.py --save-plot results/comparison.png

# See all options
python scripts/compare_optimizers.py --help
```

### Available Options

- `--dim`: Total dimensionality (default: 1000)
- `--sensitive-dims`: Number of sensitive dimensions (default: 20)
- `--landscape`: Type of landscape - 'quadratic' or 'gaussian' (default: quadratic)
- `--max-iters`: Maximum iterations (default: 500)
- `--device`: Device - 'cpu', 'cuda', 'mps', or auto (default: auto)
- `--seed`: Random seed (default: 42)
- `--init-distance`: Initial distance from optimum (default: 10.0)
- `--es-population`: ES population size (default: auto based on dimensionality)
- `--es-sigma`: ES initial step size (default: 1.0)
- `--adamw-lr`: AdamW learning rate (default: 0.01)

## Repository Structure

```
├── core/               # Core library modules
│   ├── landscapes.py   # Rotated optimization landscapes
│   ├── optimizers.py   # Evolution Strategy implementations
│   └── utils.py        # Utility functions
├── scripts/            # Standalone scripts
│   └── compare_optimizers.py  # ES vs AdamW comparison
└── notebooks/          # Jupyter notebooks
    └── ordered_stats_compute.ipynb  # Order statistics analysis
```

## Landscapes

### RotatedQuadratic
- Quadratic landscape with k sensitive and (d-k) flat dimensions
- Eigenvalues: k large (configurable range), d-k small (~1e-6)
- Random orthogonal rotation for non-axis-aligned Hessian
- Optimum at configurable location (default: origin)

### RotatedGaussian
- Negative log-likelihood of Gaussian
- k small variances (sensitive), d-k large variances (flat)
- Rotated covariance structure
- Optimum at mean of Gaussian

## Optimizers

### SeparableES
- Natural Evolution Strategy with CMA-ES style recombination
- Logarithmic weights for top performers
- Cumulative step-size adaptation (CSA)
- Population size heuristic: λ = 4 + ⌊3 log₂(d)⌋

### SimpleES
- Baseline ES with finite-difference gradient estimation
- Vanilla gradient descent updates

## Example Results

The comparison script generates plots showing:
1. **Loss convergence** - How quickly each optimizer reduces the loss
2. **Distance to optimum** - Geometric convergence towards the true minimum
3. **Gradient norm** - Magnitude of gradients during optimization
4. **Final convergence** - Detailed view of the last 20% of iterations

Typical findings:
- **ES**: More robust to flat dimensions, slower initial convergence
- **AdamW**: Faster initial progress, may struggle in very high dimensions
- Performance depends on: dimensionality, condition number, rotation

## Research Applications

This codebase supports research into:
- Behavior of evolution strategies on ill-conditioned landscapes
- Gradient-free vs gradient-based optimization comparison
- Impact of landscape geometry on optimizer performance
- Natural gradient methods and their relation to ES

## Citation

If you use this code in your research, please see CLAUDE.md for additional context.
