"""Utility functions for optimization experiments."""

import torch
import numpy as np


def get_device(device_str=None):
    """
    Get PyTorch device with automatic detection.

    Parameters
    ----------
    device_str : str, optional
        Device specification ('cpu', 'cuda', 'mps', or None for auto)

    Returns
    -------
    torch.device
        The selected device
    """
    if device_str is not None:
        return torch.device(device_str)

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def random_rotation_matrix(dim, device='cpu', dtype=torch.float32):
    """
    Generate a random orthogonal rotation matrix using QR decomposition.

    Parameters
    ----------
    dim : int
        Dimensionality of the rotation matrix
    device : torch.device or str
        Device to place the matrix on
    dtype : torch.dtype
        Data type for the matrix

    Returns
    -------
    torch.Tensor
        Random orthogonal matrix of shape (dim, dim)
    """
    # Generate random matrix
    A = torch.randn(dim, dim, device=device, dtype=dtype)

    # QR decomposition gives us an orthogonal matrix Q
    Q, R = torch.linalg.qr(A)

    # Ensure determinant is +1 (proper rotation, not reflection)
    # by flipping sign of first column if det is negative
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def distance_to_optimum(x, optimum):
    """
    Compute L2 distance to optimum.

    Parameters
    ----------
    x : torch.Tensor
        Current position
    optimum : torch.Tensor
        Optimal position

    Returns
    -------
    float
        L2 distance
    """
    return torch.norm(x - optimum).item()


def compute_metrics(x, landscape, optimum):
    """
    Compute various optimization metrics.

    Parameters
    ----------
    x : torch.Tensor
        Current position
    landscape : callable
        Landscape function
    optimum : torch.Tensor
        Known optimum position

    Returns
    -------
    dict
        Dictionary with metrics: 'loss', 'distance', 'grad_norm'
    """
    x_eval = x.detach().clone().requires_grad_(True)
    loss = landscape(x_eval)

    # Compute gradient
    loss.backward()
    grad_norm = torch.norm(x_eval.grad).item() if x_eval.grad is not None else 0.0

    return {
        'loss': loss.item(),
        'distance': distance_to_optimum(x, optimum),
        'grad_norm': grad_norm
    }
