import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import RenumberAtoms
from gnnpooling.utils.tensor_utils import is_tensor, is_numpy, one_of_k_encoding
from gnnpooling.utils import const


def to_mol(mol, addHs=False, explicitOnly=True, ordered=True, kekulize=True):
    r"""
    Convert an imput molecule (smiles representation) into a Chem.Mol
    :raises ValueError: if the input is neither a CHem.Mol nor a string

    Arguments
    ----------
        mol: str or rdkit.Chem.Mol
            SMILES of a molecule or a molecule
        addHs: bool, optional): Whether hydrogens should be added the molecule.
           (Default value = False)
        explicitOnly: bool, optional
            Whether to only add explicit hydrogen or both
            (implicit and explicit) when addHs is set to True.
            (Default value = True)
        ordered: bool, optional, default=False
            Whether the atom should be ordered. This option is important if you want to ensure
            that the features returned will always maintain a sinfle atom order for the same molecule,
            regardless of its original smiles representation
        kekulize: bool, optional, default=True
            Kekulize input molecule

    Returns
    -------
        mol: rdkit.Chem.Molecule
            the molecule if some conversion have been made.
            If the conversion fails None is returned so make sure that you handle this case on your own.
    """
    if not isinstance(mol, (str, Chem.Mol)):
        raise ValueError("Input should be a CHem.Mol or a string")
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    # make more sense to add hydrogen before ordering
    if mol is not None and addHs:
        mol = Chem.AddHs(mol, explicitOnly=explicitOnly)
    if mol and ordered:
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=True)
        new_order = sorted([(y, x) for x, y in enumerate(new_order)])
        mol = RenumberAtoms(mol, [y for (x, y) in new_order])
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol


def smiles_to_mols(mols, ordered=True):
    """Convert a list of molecules to Chem.Mol"""
    res = []
    for mol in mols:
        res.append(to_mol(mol, ordered=ordered))
    return res


def get_bond_feats(bond, enc, add_bond=False):
    # Encode bond type as a feature vector
    if not add_bond:
        return np.ones(1, dtype=int)
    bond_type = bond.GetBondType()
    for i, v in enumerate(const.BOND_TYPES):
        if v == bond_type:
            enc[i] = 1
            break
    return enc

def get_add_feats(atom, explicit_H=False, use_chirality=True):
    feats = []
    feats.extend(one_of_k_encoding(atom.GetDegree(), const.ATOM_DEGREE_LIST))
    # mplicit valence
    feats.extend(one_of_k_encoding(
        atom.GetImplicitValence(), const.IMPLICIT_VALENCE))
    # add hybridization type of atom
    feats.extend(one_of_k_encoding(
        atom.GetHybridization(), const.HYBRIDIZATION_LIST))
    # whether the atom is aromatic or not
    feats.append(int(atom.GetIsAromatic()))
    # atom formal charge
    feats.append(atom.GetFormalCharge())
    # add number of radical electrons
    feats.append(atom.GetNumRadicalElectrons())
    # atom is in ring
    feats.append(int(atom.IsInRing()))

    if not explicit_H:
        # number of hydrogene, is usually 0 after Chem.AddHs(mol) is called
        feats.extend(one_of_k_encoding(atom.GetTotalNumHs(), const.ATOM_NUM_H))

    if use_chirality:
        try:
            feats.extend(one_of_k_encoding(
                atom.GetProp('_CIPCode'), const.CHIRALITY_LIST))
            feats.append(int(atom.HasProp('_ChiralityPossible')))

        except:
            feats.extend([0, 0, int(atom.HasProp('_ChiralityPossible'))])

    return np.asarray(feats, dtype=np.float32)


class GraphTransformer():
    def __init__(self, mol_size=[0, 100], explicit_H=False, all_feat=True, add_bond=False, one_hot_bond=False, atom_list=None):
        # if this is not set, packing of graph would be expected later
        self.mol_size = mol_size
        self.n_atom_feat = 0
        self.explicit_H = explicit_H
        self.use_chirality = True
        self.all_feat = all_feat
        self.add_bond = add_bond
        self.bond_dim = 1
        self.one_hot_bond = one_hot_bond
        if add_bond:
            self.bond_dim = len(const.BOND_TYPES) + int(one_hot_bond)
        self.atom_list = atom_list or const.ATOM_LIST
        self._set_num_features()

    @staticmethod
    def atom_dim():
        return len(self.atom_list) + 1

    def _set_num_features(self):
        r"""Compute the number of features for each atom and bond
        """
        self.n_atom_feat = 0
        # add atom type required
        self.n_atom_feat += len(self.atom_list) + 1
        if self.all_feat:
            # add atom degree
            self.n_atom_feat += len(const.ATOM_DEGREE_LIST) + 1
            # add valence implicit
            self.n_atom_feat += len(const.IMPLICIT_VALENCE) + 1
            # aromatic, formal charge, radical electrons, in_ring
            self.n_atom_feat += 4
            # hybridation_list
            self.n_atom_feat += len(const.HYBRIDIZATION_LIST) + 1
            # number of hydrogen
            if not self.explicit_H:
                self.n_atom_feat += len(const.ATOM_NUM_H) + 1
            # chirality
            if self.use_chirality:
                self.n_atom_feat += 3

    def transform(self, mols, ignore_errors=False):
        features = []
        mol_list = []
        for i, ml in enumerate(mols):
            mol = None
            if ignore_errors:
                try:
                    mol = to_mol(ml, addHs=self.explicit_H, ordered=True)
                except Exception as e:
                    pass
            else:
                mol = to_mol(ml, addHs=self.explicit_H, ordered=True)

            if mol is None and not ignore_errors:
                raise (ValueError(
                    'Molecule {} cannot be transformed adjency graph'.format(ml)))
            if mol:
                num_atom = mol.GetNumAtoms()
                if (self.mol_size[0] and self.mol_size[0] > num_atom) or (
                        self.mol_size[1] and num_atom > self.mol_size[1]):
                    mol = None
            mol_list.append(mol)

        for mol in mol_list:
            feat = None
            if mol is not None:
                if ignore_errors:
                    try:
                        feat = self._transform(mol)
                    except:
                        pass
                else:
                    feat = self._transform(mol)
            features.append(feat)
        return features

    def _transform(self, mol):
        if mol is None:
            raise ValueError("Expecting a Chem.Mol object, got None")

        n_atoms = mol.GetNumAtoms()
        # for each atom, we would have one neighbor at each of its valence state
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=np.int)
        adj_matrix = np.zeros(
            (n_atoms, n_atoms, self.bond_dim), dtype=np.int)
        atom_arrays = []
        atom_add_feat_arrays = []
        for a_idx in range(0, min(n_atoms, mol.GetNumAtoms())):
            atom = mol.GetAtomWithIdx(a_idx)
            # pos = ((const.ATOM_LIST + [atom.GetSymbol()]).index(atom.GetSymbol()) +1 ) % (1+len(const.ATOM_LIST))
            # atom_arrays.append([pos])
            atom_feats = one_of_k_encoding(atom.GetSymbol(), self.atom_list)
            if self.all_feat:
                atom_add_feat_arrays.append(
                    get_add_feats(atom, explicit_H=self.explicit_H, use_chirality=self.use_chirality))
            atom_arrays.append(atom_feats)
            for n_pos, neighbor in enumerate(atom.GetNeighbors()):
                n_idx = neighbor.GetIdx()
                # do not exceed hard limit on the maximum number of atoms
                # allowed
                bond = mol.GetBondBetweenAtoms(a_idx, n_idx)
                bond_feat = get_bond_feats(bond, np.zeros(self.bond_dim, dtype=np.int), self.add_bond)
                if n_idx < n_atoms:
                    adj_matrix[n_idx, a_idx] = bond_feat
                    adj_matrix[a_idx, n_idx] = bond_feat

                # keep self loop to empty, then rewrite graph conv

        bond_one_hot = np.zeros(self.bond_dim, dtype=np.int)
        if self.add_bond and self.one_hot_bond:
            bond_one_hot[-1] = 1
            adj_matrix[adj_matrix.sum(axis=-1)==0] = bond_one_hot
        n_atom_shape = len(atom_arrays[0])
        atom_matrix = np.zeros(
            (n_atoms, n_atom_shape)).astype(np.int)
        for idx, atom_array in enumerate(atom_arrays):
            atom_matrix[idx, :] = atom_array
        if atom_add_feat_arrays and self.all_feat:
            feat_matrix = np.concatenate(
                [atom_matrix, np.asarray(atom_add_feat_arrays)], axis=1)
            return (adj_matrix, feat_matrix)
        return adj_matrix, atom_matrix

    def __call__(self, mols, dtype=np.float, cuda=False, **kwargs):
        feats = self.transform(mols, **kwargs)
        ids = []
        for f_id, feat in enumerate(feats):
            if feat is not None:
                ids.append(f_id)
        graphs = list(filter(None.__ne__, feats))

        if is_tensor(dtype):
            graphs = [(to_tensor(x1, gpu=cuda, dtype=dtype), to_tensor(
                x2, gpu=cuda, dtype=dtype)) for (x1, x2) in graphs]
        elif is_numpy(dtype):
            graphs = [[np.array(x, dtype=dtype) for x in y] for y in graphs]
        else:
            raise (TypeError('The type {} is not supported'.format(dtype)))
        return graphs, ids
