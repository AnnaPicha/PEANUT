import torch
import numpy as np
from e3nn.o3 import spherical_harmonics



def test_spherical_harmonics_l1_directions():
    # Directions along the axes
    number_atoms=10
    coordinates = torch.randn(number_atoms,3)
    
    l = 7

    # Compute spherical harmonics
    Y = spherical_harmonics(l, coordinates, normalization='component', normalize=True)

    Y_real = Y.real   # just real part
    assert Y_real.shape == (number_atoms, 2 * l + 1)
    assert torch.any(torch.abs(Y_real) > 1e-5)
    # Check that each row has non-zero magnitude
    row_norms = torch.linalg.norm(Y_real, dim=1)
    assert torch.all(row_norms > 1e-5), "Some SH rows are essentially zero"
    assert Y_real.dtype in [torch.float32, torch.float64]
    assert torch.isfinite(Y_real).all(), "Some SH values are NaN or Inf"
    delta = torch.tensor([1e-5, 0.0, 0.0])

    Y_rotated = spherical_harmonics(l, coordinates + delta, normalization='component', normalize=True).real
    diff = torch.abs(Y_real - Y_rotated)
    assert torch.all(diff < 1e-3), "SH changed too much for tiny perturbation"

