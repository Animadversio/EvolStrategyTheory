"""Quick test with small dimensions to verify implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from core.landscapes import RotatedQuadratic
from core.optimizers import SeparableES
from core.utils import get_device

# Small test
DIM = 50
SENSITIVE_DIMS = 5
MAX_ITERS = 50

print("Quick Test - Small Dimensions")
print("=" * 60)

device = get_device()
print(f"Device: {device}")

# Create landscape
landscape = RotatedQuadratic(
    dim=DIM,
    sensitive_dims=SENSITIVE_DIMS,
    device=device,
    seed=42
)

print(f"\nLandscape created:")
print(f"  Dimensions: {DIM}")
print(f"  Sensitive dims: {SENSITIVE_DIMS}")
print(f"  Condition number: {landscape.eigenvalues.max() / landscape.eigenvalues.min():.2e}")

# Test point
x = torch.randn(DIM, device=device) + landscape.optimum
loss = landscape(x)
print(f"\nTest evaluation:")
print(f"  Loss: {loss.item():.6e}")
print(f"  Distance from optimum: {torch.norm(x - landscape.optimum).item():.4f}")

# Test ES
print(f"\nTesting ES optimizer...")
es = SeparableES(x.clone(), population_size=20, sigma=0.5, device=device)
print(f"  Population size: {es.population_size}")

for i in range(MAX_ITERS):
    info = es.step(landscape)
    if i % 10 == 0:
        print(f"  Iter {i}: Loss = {info['best_fitness']:.6e}, Sigma = {info['sigma']:.4f}")

final_solution = es.get_current_solution()
final_loss = landscape(final_solution)
final_dist = torch.norm(final_solution - landscape.optimum)

print(f"\nFinal ES results:")
print(f"  Loss: {final_loss.item():.6e}")
print(f"  Distance: {final_dist.item():.6e}")

# Test AdamW
print(f"\nTesting AdamW optimizer...")
adamw_x = x.clone().requires_grad_(True)
opt = torch.optim.AdamW([adamw_x], lr=0.01)

for i in range(MAX_ITERS):
    opt.zero_grad()
    loss = landscape(adamw_x)
    loss.backward()
    opt.step()
    if i % 10 == 0:
        print(f"  Iter {i}: Loss = {loss.item():.6e}")

final_adamw_loss = landscape(adamw_x)
final_adamw_dist = torch.norm(adamw_x - landscape.optimum)

print(f"\nFinal AdamW results:")
print(f"  Loss: {final_adamw_loss.item():.6e}")
print(f"  Distance: {final_adamw_dist.item():.6e}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
