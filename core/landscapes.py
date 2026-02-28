"""Optimization landscape classes with rotated coordinate systems."""

import torch
import numpy as np
from .utils import random_rotation_matrix


class RotatedQuadratic:
    """
    Rotated quadratic landscape with k sensitive and (d-k) flat dimensions.

    The landscape is constructed as:
    f(x) = 0.5 * (x - x_opt)^T H (x - x_opt)

    where H = R^T diag(eigenvalues) R, with R being a random rotation matrix.

    Parameters
    ----------
    dim : int
        Total dimensionality
    sensitive_dims : int
        Number of sensitive dimensions (k)
    eigenvalue_range : tuple
        (min_eigenvalue, max_eigenvalue) for sensitive dimensions
    flat_eigenvalue : float
        Eigenvalue for flat dimensions (should be small, e.g., 1e-6)
    optimum : torch.Tensor or None
        Location of minimum. If None, set to zeros.
    device : torch.device or str
        Device to place tensors on
    seed : int or None
        Random seed for reproducibility
    """

    def __init__(self, dim=1000, sensitive_dims=10,
                 eigenvalue_range=(1.0, 100.0),
                 flat_eigenvalue=1e-6,
                 optimum=None,
                 device='cpu',
                 seed=None):
        self.dim = dim
        self.sensitive_dims = sensitive_dims
        self.device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create eigenvalues: k large (sensitive), d-k small (flat)
        eigenvalues = torch.ones(dim, device=self.device) * flat_eigenvalue
        # Randomly distribute sensitive eigenvalues
        sensitive_eigs = torch.logspace(
            np.log10(eigenvalue_range[0]),
            np.log10(eigenvalue_range[1]),
            sensitive_dims,
            device=self.device
        )
        sensitive_eigs, _ = torch.sort(sensitive_eigs, descending=True)
        eigenvalues[:sensitive_dims] = sensitive_eigs

        # Create random rotation matrix
        self.rotation = random_rotation_matrix(dim, device=self.device)

        # Construct Hessian: H = R^T diag(eigs) R
        D = torch.diag(eigenvalues)
        self.hessian = self.rotation.T @ D @ self.rotation

        # Set optimum
        if optimum is None:
            self.optimum = torch.zeros(dim, device=self.device)
        else:
            self.optimum = optimum.to(self.device)

        # Store eigenvalues for reference
        self.eigenvalues = eigenvalues

    def __call__(self, x):
        """
        Evaluate the landscape at position x.

        Parameters
        ----------
        x : torch.Tensor
            Position vector of shape (dim,)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        diff = x - self.optimum
        return 0.5 * diff @ self.hessian @ diff

    def gradient(self, x):
        """
        Compute gradient analytically.

        Parameters
        ----------
        x : torch.Tensor
            Position vector

        Returns
        -------
        torch.Tensor
            Gradient vector
        """
        diff = x - self.optimum
        return self.hessian @ diff

    def project_to_sensitive(self, x):
        """
        Project point to sensitive subspace (for visualization).

        Parameters
        ----------
        x : torch.Tensor
            Position vector

        Returns
        -------
        torch.Tensor
            Projection onto first k sensitive dimensions in rotated space
        """
        # Transform to eigenbasis
        x_rotated = self.rotation @ (x - self.optimum)
        return x_rotated[:self.sensitive_dims]


class RotatedGaussian:
    """
    Negative log-likelihood of Gaussian mixture in rotated space.

    Landscape is:
    f(x) = -log(sum_i w_i * N(x | mu_i, Sigma_i))

    For simplicity, we use a single Gaussian centered at optimum with
    rotated anisotropic covariance.

    Parameters
    ----------
    dim : int
        Total dimensionality
    sensitive_dims : int
        Number of sensitive dimensions
    variance_range : tuple
        (min_variance, max_variance) for sensitive dimensions
    flat_variance : float
        Variance for flat dimensions (should be large for flatness)
    optimum : torch.Tensor or None
        Location of minimum (mean of Gaussian)
    device : torch.device or str
        Device to place tensors on
    seed : int or None
        Random seed for reproducibility
    """

    def __init__(self, dim=1000, sensitive_dims=10,
                 variance_range=(0.01, 1.0),
                 flat_variance=100.0,
                 optimum=None,
                 device='cpu',
                 seed=None):
        self.dim = dim
        self.sensitive_dims = sensitive_dims
        self.device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create variances: k small (sensitive), d-k large (flat)
        variances = torch.ones(dim, device=self.device) * flat_variance
        sensitive_vars = torch.logspace(
            np.log10(variance_range[0]),
            np.log10(variance_range[1]),
            sensitive_dims,
            device=self.device
        )
        variances[:sensitive_dims] = sensitive_vars

        # Create random rotation
        self.rotation = random_rotation_matrix(dim, device=self.device)

        # Construct covariance: Sigma = R^T diag(vars) R
        D = torch.diag(variances)
        self.covariance = self.rotation.T @ D @ self.rotation

        # Compute inverse and log determinant for NLL
        self.precision = torch.inverse(self.covariance)
        self.log_det = torch.logdet(self.covariance)

        # Set optimum (mean of Gaussian)
        if optimum is None:
            self.optimum = torch.zeros(dim, device=self.device)
        else:
            self.optimum = optimum.to(self.device)

        self.variances = variances

    def __call__(self, x):
        """
        Evaluate negative log-likelihood at position x.

        Parameters
        ----------
        x : torch.Tensor
            Position vector

        Returns
        -------
        torch.Tensor
            Scalar NLL value
        """
        diff = x - self.optimum
        # NLL = 0.5 * (log det Sigma + (x-mu)^T Sigma^-1 (x-mu) + d*log(2*pi))
        # We drop the constant term
        mahalanobis = diff @ self.precision @ diff
        return 0.5 * (self.log_det + mahalanobis)

    def gradient(self, x):
        """
        Compute gradient analytically.

        Parameters
        ----------
        x : torch.Tensor
            Position vector

        Returns
        -------
        torch.Tensor
            Gradient vector
        """
        diff = x - self.optimum
        return self.precision @ diff

    def project_to_sensitive(self, x):
        """
        Project point to sensitive subspace.

        Parameters
        ----------
        x : torch.Tensor
            Position vector

        Returns
        -------
        torch.Tensor
            Projection onto first k sensitive dimensions in rotated space
        """
        x_rotated = self.rotation @ (x - self.optimum)
        return x_rotated[:self.sensitive_dims]
