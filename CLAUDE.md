# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research codebase for analyzing evolutionary strategy theory, specifically focused on order statistics of Gaussian distributions as applied to CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

## Project Structure

- `core/` - Core library modules (PyTorch-based)
  - `landscapes.py` - Rotated optimization landscapes (quadratic, Gaussian)
  - `optimizers.py` - Evolution Strategy implementations (SeparableES, SimpleES)
  - `utils.py` - Utility functions (device management, metrics, rotation matrices)
- `scripts/` - Standalone analysis scripts
  - `compare_optimizers.py` - CLI tool for ES vs AdamW comparison
  - `demo_comparison.py` - Interactive demo script (no CLI, easy to modify)
  - `quick_test.py` - Small-scale test script for verification
- `notebooks/` - Jupyter notebooks for research analysis
  - `ordered_stats_compute.ipynb` - Order statistics computation and CMA-ES analysis

## Key Research Components

### Order Statistics Analysis

The primary research focuses on computing exact moments (means, covariances) of Gaussian order statistics Z_{i:λ} where Z_k ~ N(0,1) are i.i.d. samples:

1. **Gaussian Order Statistic PDF**: Computes the probability density function for the i-th order statistic from λ samples
2. **Mean Computation**: Two methods for computing E[Z_{i:λ}]:
   - Direct z-space integration over (-∞, ∞)
   - U-space integration using the transformation U = Φ(Z) where U_{i:λ} ~ Beta(i, λ+1-i) (more stable)
3. **Joint Moments**: Computes covariances between order statistics using 2D integration over joint distributions

### CMA-ES Weight Analysis

The notebook analyzes CMA-ES recombination weights and their geometric properties:

- **Standard CMA-ES weights**: Logarithmic weights w_i = log(μ + 0.5) - log(i) for top μ = λ/2 individuals
- **On-manifold shift**: Weighted mean of order statistics (gradient alignment component)
- **Orthogonal variance**: Variance in directions perpendicular to gradient
- **Cosine with gradient**: Measures alignment of the natural gradient estimate with the true gradient across different dimensionalities

The analysis reveals how population size and dimensionality affect the geometric properties of CMA-ES gradient estimates.

## Optimization Landscape Experiments

### Core Library Components

**Landscapes** (`core/landscapes.py`):
- `RotatedQuadratic`: Quadratic landscape with k sensitive and (d-k) flat dimensions
  - Hessian: H = R^T diag(eigenvalues) R where R is random rotation
  - k large eigenvalues (sensitive), d-k small eigenvalues (flat)
  - Non-axis-aligned via random orthogonal rotation
- `RotatedGaussian`: Negative log-likelihood of Gaussian with rotated covariance
  - k small variances (sensitive), d-k large variances (flat)

**Optimizers** (`core/optimizers.py`):
- `SeparableES`: Natural ES with CMA-ES recombination weights and step-size adaptation (CSA)
- `SimpleES`: Baseline ES with finite-difference gradient estimation

### Running Experiments

**Quick Test** (verify installation):
```bash
mamba activate torch
python scripts/quick_test.py
```

**Interactive Demo** (recommended for exploration):
```bash
mamba activate torch
python scripts/demo_comparison.py
```
Modify parameters directly in `demo_comparison.py` (DIM, SENSITIVE_DIMS, etc.). Results are in global scope for post-analysis.

**CLI Comparison Tool** (for systematic experiments):
```bash
mamba activate torch
python scripts/compare_optimizers.py --dim 1000 --sensitive-dims 20 --max-iters 500
```

### Key Parameters

- `dim`: Total dimensionality (typical: 100-10000)
- `sensitive_dims`: Number of sensitive dimensions k where k << d
- `landscape`: 'quadratic' or 'gaussian'
- `eigenvalue_range`: Range for sensitive eigenvalues (affects condition number)
- `flat_eigenvalue`: Eigenvalue for flat dimensions (typically 1e-6)

### Typical Use Cases

1. **Compare ES vs gradient-based on ill-conditioned problems**: High dimensionality (d=1000+), few sensitive dims (k=5-20)
2. **Study effect of rotation**: Compare axis-aligned vs rotated landscapes
3. **Analyze step-size adaptation**: Track ES sigma over iterations
4. **Evaluate scalability**: Vary d and k to understand performance

## Working with Notebooks

### Running the Main Notebook

```bash
jupyter notebook notebooks/ordered_stats_compute.ipynb
```

or use VS Code's Jupyter integration.

### Key Functions

Located in `notebooks/ordered_stats_compute.ipynb`:

- `gaussian_orderstat_pdf(z, lam, i)` - PDF of Z_{i:λ}
- `gaussian_orderstat_mean_u_integral(lam, i)` - Compute E[Z_{i:λ}] (recommended method)
- `gaussian_orderstat_means(lam, method="u")` - Compute all order statistic means for λ samples
- `moments_gaussian_orderstats_analytic(lam, i, j)` - Compute E[Z_i], E[Z_j], and Cov(Z_i, Z_j)
- `moments_gaussian_orderstats_mc(lam, i, j, n_mc)` - Monte Carlo validation
- `cmaes_weights(population_size)` - Generate CMA-ES recombination weights

## Environment Setup

**IMPORTANT**: Always use the `torch` mamba environment for running code:

```bash
mamba activate torch
python scripts/demo_comparison.py
```

All Python code in this repository should be run with `mamba activate torch` first.

## Dependencies

The code uses standard scientific Python libraries:
- `torch` - Neural network library (used for optimization and tensor operations)
- `numpy` - Array operations
- `scipy` - Integration (`scipy.integrate`) and distributions (`scipy.stats`)
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation for results tables

Install all dependencies: `pip install -r requirements.txt`

## Development Notes

- The U-space integration method (`gaussian_orderstat_mean_u_integral`) is preferred over Z-space integration for numerical stability
- Integration tolerances can be adjusted via `epsabs` and `epsrel` parameters for accuracy/speed tradeoffs
- Monte Carlo validation is provided to verify analytical computations
- The CMA-ES population size follows the heuristic: λ = 4 + floor(3 * log₂(d)) where d is dimensionality