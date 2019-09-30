import torch
from rdkit import Chem 

ATOM_LIST = ['C', 'N', 'O', 'F']
# BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#               Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

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
