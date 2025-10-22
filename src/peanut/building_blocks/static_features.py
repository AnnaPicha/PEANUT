import torch
import numpy as np
from e3nn.o3 import spherical_harmonics



class Static_Message_Features():
    
    """
    Class to compute static message features based on spherical harmonics and radial symmetry functions.
    Args:
        vectors (torch.Tensor): Tensor of shape (num_atom_pairs, 3) representing the vectors between atom pairs.
        spherical_harmonic_l (int): Order of the spherical harmonics to be computed.
        symmetry_functions_b (int): Number of radial symmetry functions to be used.
    Returns:
        None
    """

    def __init__(self,
                 vectors: torch.Tensor,     # shape (num_atom_pairs, 3)
                 spherical_harmonic_l: int, # order of the spherical harmonics
                 symmetry_functions_b: int  # number of radial symmetry functions
                 ):
        
        self.vectors = vectors
        self.spherical_harmonic_l = spherical_harmonic_l
        self.symmetry_functions_b = symmetry_functions_b    


    def get_spherical_harmonics(l: int, 
                                 coordinates: torch.Tensor
                                 ) -> torch.Tensor: # shape (num_atom_pairs, 2l+1)
        """ 
        Computes the spherical harmonics for given coordinates and order l.
        Args:
            l (int): Order of the spherical harmonics.
            coordinates (torch.Tensor): Tensor of shape (num_atom_pairs, 3) representing the vectors.
        Returns:
            torch.Tensor: Real part of the spherical harmonics of shape (num_atom_pairs, 2l+1).
        """

        # Compute spherical harmonics
        Y = spherical_harmonics(l, coordinates, normalization='component', normalize=True)

        Y_real = Y.real   # just real part
        return Y_real
    

    def get_radial_symmetry_functions(vectors: torch.Tensor,
                                      b: int
                                      ) -> torch.Tensor: # shape (num_atom_pairs, b)
        """
        Computes radial symmetry functions for given vectors.
        Args:
            vectors (torch.Tensor): Tensor of shape (num_atom_pairs, 3) representing the vectors.
            b (int): Number of radial symmetry functions.
        Returns:
            torch.Tensor: Radial symmetry functions of shape (num_atom_pairs, b).
        """

        # Compute distances
        distances = torch.linalg.norm(vectors, dim=1, keepdim=True)  # shape (num_atom_pairs, 1)    
        # Create radial symmetry functions (example: Gaussian functions)
        # radial_sym_funcs = torch.exp(-((distances - torch.linspace(0, 5, b).to(distances.device))**2) / (2 * 0.5**2))  # shape (num_atom_pairs, b)
        # return radial_sym_funcs