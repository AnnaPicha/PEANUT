# import pandas as pd
import numpy as np
import h5py
from ase.data import chemical_symbols
from ase import Atoms, units
from ase.io import write,read
import os
import tqdm
import random
from rdkit import Chem


def format_data():
    # path = '../../../data/'
    filename = f'a_wpS'
    # data = h5py.File(f"{filename}.hdf5", "r")

    # Pfad relativ zum Repo-Root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    data_file = os.path.join(repo_root, 'data', f'{filename}.hdf5')
    xyz_file = os.path.join(repo_root, 'data', f'{filename}.xyz')

    data = h5py.File(data_file, 'r')

    allowed_symbols = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}

    for key in tqdm.tqdm(data.keys()):

        Z = data[key]['atomic_numbers'][:]
        symbols = [chemical_symbols[int(z)] for z in Z]
        symbols_set = set(symbols)

        is_valid = symbols_set.issubset(allowed_symbols)
        # only use molecules with allowed symbols
        if is_valid:
            try:
                mol = Chem.MolFromSmiles(data[key]['smiles'][0].decode('utf-8'))
                charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
                # only use neutral molecules
                if np.sum(charges) == 0:
                    for i in range(len(data[key]['conformations'])):
                        pos_bohr = data[key]['conformations'][i]          # Bohr
                        E_hartree_total = data[key]['dft_total_energy'][i]          # Hartree
                        E_hartree = data[key]['formation_energy'][i]          # Hartree
                        grad_hartree_per_bohr = data[key]['dft_total_gradient'][i]  # Hartree/Bohr

                        pos_ang = pos_bohr * units.Bohr
                        E_eV = E_hartree * units.Hartree
                        E_ev_total = E_hartree_total * units.Hartree
                        forces_eV_per_A = -grad_hartree_per_bohr * (units.Hartree / units.Bohr) # must be negative!!!!!!!

                        atoms = Atoms(symbols=symbols, positions=pos_ang)
                        atoms.info['energy'] = E_eV
                        atoms.info['total_energy'] = E_ev_total
                        atoms.arrays['forces'] = forces_eV_per_A

                        mode = "w" if not os.path.exists(f"{filename}.xyz") else "a"
                        atoms.write(f"{xyz_file}", append=True)
            except:
                print(f"Could not process {key} with SMILES {data[key]['smiles'][0].decode('utf-8')}")
