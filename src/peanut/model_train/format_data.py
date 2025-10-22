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
import torch
# import os
import torch
from extxyz import read  # korrekt importieren



def format_data_xyz(repo_root:str, dataset: str):


    data_file = os.path.join(repo_root, 'data', f'{dataset}.hdf5')
    xyz_file = os.path.join(repo_root, 'data', f'{dataset}.xyz')

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

                        # mode = "w" if not os.path.exists(f"{dataset}.xyz") else "a"
                        atoms.write(f"{xyz_file}", append=True)
            except:
                print(f"Could not process {key} with SMILES {data[key]['smiles'][0].decode('utf-8')}")







# import torch
# from extxyz import io
# import os

# def xyz_to_pt(xyz_file_name, output_file=None):
#     """
#     Liest eine extxyz/xyz Datei ein und speichert sie als PyTorch .pt Datei.
    
#     Args:
#         xyz_file (str): Pfad zur xyz/extxyz Datei.
#         output_file (str): Pfad für das gespeicherte .pt File. Default: xyz_file mit .pt Endung.
#     """
#     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
#     # data_file = os.path.join(repo_root, 'data', f'{dataset}.hdf5')
#     xyz_file = os.path.join(repo_root, 'data', f'{xyz_file_name}.xyz')
#     if output_file is None:
#         output_file = os.path.splitext(xyz_file_name)[0] + ".pt"

#     dataset = []

#     # Datei einlesen
#     with io.read(xyz_file) as f:
#         for atoms in f:
#             energy = atoms.info.get("energy", None)
#             total_energy = atoms.info.get("total_energy", None)
#             forces = atoms.arrays.get("forces", None)

#             if energy is not None and forces is not None:
#                 data = {
#                     "symbols": atoms.get_chemical_symbols(),
#                     "positions": atoms.get_positions(),
#                     "energy": energy,
#                     "total_energy": total_energy,
#                     "forces": forces
#                 }
#                 dataset.append(data)
#             else:
#                 print(f"Skipping molecule: missing energy or forces. Info keys: {atoms.info.keys()}, Arrays: {atoms.arrays.keys()}")

#     # Speichern
#     torch.save(dataset, output_file)
#     print(f"Saved {len(dataset)} molecules to {output_file}")


# # if __name__ == "__main__":
# #     xyz_path = "data/a_wpS.xyz"  # Pfad zu deiner Datei
# #     xyz_to_pt(xyz_path)



def xyz_to_pt(repo_root: str, xyz_file_name: str, output_file=None):
    """
    Liest eine extxyz/xyz Datei ein und speichert sie als PyTorch .pt Datei.

    Args:
        xyz_file_name (str): Name der xyz/extxyz Datei (ohne Pfad, ohne Endung).
        output_file (str): Pfad für das gespeicherte .pt File. Default: data/{xyz_file_name}.pt im Repo-Root.
    """
    # Repo-Root bestimmen
    # repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    xyz_file = os.path.join(repo_root, 'data', f'{xyz_file_name}.xyz')
    
    if output_file is None:
        output_file = os.path.join(repo_root, 'data', f'{xyz_file_name}.pt')

    dataset = []

    try:
        frames = read(xyz_file)  # alle Frames einlesen
    except Exception as e:
        print(f"Fehler beim Einlesen der xyz-Datei: {e}")
        return

    for atoms in frames:
        energy = atoms.info.get("energy", None)
        total_energy = atoms.info.get("total_energy", None)
        forces = atoms.arrays.get("forces", None)

        if energy is not None and forces is not None:
            data = {
                "symbols": atoms.get_chemical_symbols(),
                "positions": atoms.get_positions(),
                "energy": energy,
                "total_energy": total_energy,
                "forces": forces
            }
            dataset.append(data)
        else:
            print(f"Skipping molecule: missing energy or forces. Info keys: {atoms.info.keys()}, Arrays: {atoms.arrays.keys()}")

    torch.save(dataset, output_file)
    print(f"Saved {len(dataset)} molecules to {output_file}")


def format_data_from_hdf5(dataset: str):

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

    xyz_file = os.path.join(repo_root, 'data', f'{dataset}.xyz')
    pt_file = os.path.join(repo_root, 'data', f'{dataset}.pt')

    # Prüfen, ob die Dateien schon existieren
    if os.path.exists(xyz_file) or os.path.exists(pt_file):
        print('#' * 100)
        raise FileExistsError(f"ERROR: '{xyz_file}' or '{pt_file}' already exists! Aborting.")


    format_data_xyz(repo_root=repo_root, dataset=dataset)
    xyz_to_pt(repo_root=repo_root, xyz_file_name=dataset)

    
    # data = h5py.File(data_file, 'r')
    # return data
# if __name__ == "__main__":
#     xyz_to_pt("a_wpS")
