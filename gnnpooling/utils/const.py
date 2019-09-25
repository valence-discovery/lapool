import torch
from rdkit import Chem 

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B',
             'K', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Cu', 'Ni', 'Cd', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg']
# BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#               Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

# ATOM_LIST = ['Sn', 'I', 'Al', 'Zr', 'P', 'Cl', 'Ag', 'Ge', 'Cr',
#         'Ba', 'Gd', 'Mn', 'Dy', 'H', 'S', 'K', 'Mo', 'Se',
#         'As', 'Fe', 'Co', 'Sb', 'Sr', 'O', 'Au', 'Ni', 'Na',
#         'Bi', 'Cd', 'In', 'Mg', 'V', 'Ti', 'Tl', 'B', 'Cu',
#         'F', 'Pt', 'Pd', 'Br', 'Ca', 'C', 'Yb', 'N', 'Pb',
#         'Be', 'Li', 'Zn', 'Si', 'Hg', 'Nd']

# ATOM_LIST = ['I', 'Al', 'P', 'Cl', 'Mn', 'S', 'K', 'Fe', 'O', 'Ni', 'Na', 'Mg', 'B',
#         'F', 'Br', 'Ca', 'C', 'N', 'Be', 'Li', 'Zn', 'Hg']

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

ATOM_NUM_H = [0, 1, 2, 3, 4]
IMPLICIT_VALENCE = [0, 1, 2, 3, 4, 5, 6]
CHARGE_LIST = [-3, -2, -1, 0, 1, 2, 3]
RADICAL_E_LIST = [0, 1, 2]
HYBRIDIZATION_LIST = [Chem.rdchem.HybridizationType.names[k] for k in sorted(
    Chem.rdchem.HybridizationType.names.keys(), reverse=True) if k != "OTHER"]
ATOM_DEGREE_LIST = range(5)
CHIRALITY_LIST = ['R']  # alternative is just S
BOND_STEREO = [0, 1, 2, 3, 4, 5]

CUDA_OK = False
if torch.cuda.is_available():
    CUDA_OK = True
