"""Evolution Strategy optimizers."""

import torch
import numpy as np


class SeparableES:
    """
    Separable Natural Evolution Strategy with CMA-ES style recombination weights.

    This implements a simplified version of CMA-ES that maintains separate
    step sizes per dimension but uses natural gradient updates.

    Parameters
    ----------
    init_params : torch.Tensor
        Initial parameter vector
    population_size : int
        Number of offspring per generation (lambda)
    sigma : float
        Initial global step size
    lr : float
        Learning rate for mean update
    sigma_lr : float
        Learning rate for step size adaptation
    device : torch.device or str
        Device to run on
    """

    def __init__(self, init_params, population_size=None, sigma=1.0,
                 lr=1.0, sigma_lr=0.1, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mean = init_params.clone().to(self.device)
        self.dim = len(self.mean)

        # Set population size using CMA-ES heuristic if not provided
        if population_size is None:
            self.population_size = int(4 + np.floor(3 * np.log(self.dim)))
        else:
            self.population_size = population_size

        self.sigma = sigma
        self.lr = lr
        self.sigma_lr = sigma_lr

        # Number of parents (top performers to use for recombination)
        self.mu = self.population_size // 2

        # CMA-ES style logarithmic weights for top mu individuals
        ranks = torch.arange(1, self.mu + 1, dtype=torch.float32, device=self.device)
        weights = torch.log(torch.tensor(self.mu + 0.5, device=self.device)) - torch.log(ranks)
        self.weights = weights / weights.sum()  # Normalize to sum to 1

        # Effective variance (for adaptive step size)
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # Evolution path for step size adaptation (cumulative)
        self.ps = torch.zeros(self.dim, device=self.device)
        self.cs = 0.3  # Cumulation constant for step size

        # Expected norm of N(0, I)
        self.chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        # Damping parameter for step size
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        self.generation = 0

    def ask(self):
        """
        Sample a population of candidate solutions.

        Returns
        -------
        population : torch.Tensor
            Population of shape (population_size, dim)
        perturbations : torch.Tensor
            Standardized perturbations used (for tell step)
        """
        # Sample standardized perturbations
        perturbations = torch.randn(self.population_size, self.dim, device=self.device)

        # Create population: mean + sigma * perturbations
        population = self.mean.unsqueeze(0) + self.sigma * perturbations

        return population, perturbations

    def tell(self, perturbations, fitness_values):
        """
        Update distribution parameters based on fitness evaluations.

        Parameters
        ----------
        perturbations : torch.Tensor
            Standardized perturbations from ask() step
        fitness_values : torch.Tensor
            Fitness values for each candidate (lower is better)
        """
        # Rank solutions (ascending order, best first)
        indices = torch.argsort(fitness_values)

        # Select top mu perturbations
        elite_indices = indices[:self.mu]
        elite_perturbations = perturbations[elite_indices]

        # Weighted recombination of elite perturbations
        weighted_perturbation = (self.weights.unsqueeze(1) * elite_perturbations).sum(dim=0)

        # Update mean using natural gradient
        self.mean = self.mean + self.lr * self.sigma * weighted_perturbation

        # Update evolution path for step size
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * weighted_perturbation

        # Adapt step size using cumulative step-size adaptation (CSA)
        norm_ps = torch.norm(self.ps).item()
        self.sigma = self.sigma * np.exp((self.sigma_lr / self.damps) * (norm_ps / self.chi_n - 1))

        self.generation += 1

    def step(self, fitness_fn):
        """
        Perform one complete generation: ask, evaluate, tell.

        Parameters
        ----------
        fitness_fn : callable
            Function that takes a parameter vector and returns scalar fitness

        Returns
        -------
        dict
            Dictionary with 'best_fitness', 'mean_fitness', 'sigma'
        """
        # Sample population
        population, perturbations = self.ask()

        # Evaluate fitness
        fitness_values = torch.tensor([fitness_fn(x).item() for x in population], device=self.device)

        # Update distribution
        self.tell(perturbations, fitness_values)

        return {
            'best_fitness': fitness_values.min().item(),
            'mean_fitness': fitness_values.mean().item(),
            'sigma': self.sigma,
            'generation': self.generation
        }

    def get_current_solution(self):
        """Get the current mean (best estimate of optimum)."""
        return self.mean.clone()

    def get_state(self):
        """Get full optimizer state."""
        return {
            'mean': self.mean.clone(),
            'sigma': self.sigma,
            'ps': self.ps.clone(),
            'generation': self.generation
        }


class SimpleES:
    """
    Simple Evolution Strategy with gradient estimation via finite differences.

    This is a baseline ES that estimates gradients by finite differences
    and updates parameters using vanilla gradient descent.

    Parameters
    ----------
    init_params : torch.Tensor
        Initial parameter vector
    population_size : int
        Number of perturbations for gradient estimation
    sigma : float
        Noise standard deviation for perturbations
    lr : float
        Learning rate for parameter updates
    device : torch.device or str
        Device to run on
    """

    def __init__(self, init_params, population_size=20, sigma=0.1, lr=0.01, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.params = init_params.clone().to(self.device)
        self.dim = len(self.params)
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.generation = 0

    def step(self, fitness_fn):
        """
        Perform one optimization step.

        Parameters
        ----------
        fitness_fn : callable
            Function that takes a parameter vector and returns scalar fitness

        Returns
        -------
        dict
            Dictionary with fitness statistics
        """
        # Sample perturbations
        noise = torch.randn(self.population_size, self.dim, device=self.device)

        # Evaluate fitness for positive and negative perturbations
        fitness_values = []
        for i in range(self.population_size):
            f_plus = fitness_fn(self.params + self.sigma * noise[i])
            f_minus = fitness_fn(self.params - self.sigma * noise[i])
            fitness_values.append((f_plus + f_minus) / 2)

            # Estimate gradient contribution from this perturbation pair
            if i == 0:
                grad_estimate = (f_plus - f_minus) * noise[i]
            else:
                grad_estimate += (f_plus - f_minus) * noise[i]

        # Average gradient estimate
        grad_estimate = grad_estimate / (2 * self.sigma * self.population_size)

        # Update parameters (gradient descent)
        self.params = self.params - self.lr * grad_estimate

        self.generation += 1

        fitness_values = torch.tensor(fitness_values, device=self.device)
        return {
            'best_fitness': fitness_values.min().item(),
            'mean_fitness': fitness_values.mean().item(),
            'sigma': self.sigma,
            'generation': self.generation
        }

    def get_current_solution(self):
        """Get current parameter vector."""
        return self.params.clone()

    def get_state(self):
        """Get optimizer state."""
        return {
            'params': self.params.clone(),
            'sigma': self.sigma,
            'generation': self.generation
        }
