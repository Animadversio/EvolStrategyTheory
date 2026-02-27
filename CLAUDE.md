# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research codebase for analyzing evolutionary strategy theory, specifically focused on order statistics of Gaussian distributions as applied to CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

## Project Structure

- `notebooks/` - Jupyter notebooks containing computational experiments and analysis
  - `ordered_stats_compute.ipynb` - Main notebook computing order statistics and analyzing CMA-ES behavior
- `core/` - (Currently empty) Intended for core library functions
- `scripts/` - (Currently empty) Intended for standalone scripts

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

## Dependencies

The code uses standard scientific Python libraries:
- `numpy` - Array operations
- `scipy` - Integration (`scipy.integrate`) and distributions (`scipy.stats`)
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation for results tables

## Development Notes

- The U-space integration method (`gaussian_orderstat_mean_u_integral`) is preferred over Z-space integration for numerical stability
- Integration tolerances can be adjusted via `epsabs` and `epsrel` parameters for accuracy/speed tradeoffs
- Monte Carlo validation is provided to verify analytical computations
- The CMA-ES population size follows the heuristic: λ = 4 + floor(3 * log₂(d)) where d is dimensionality