import numpy as np
import torch
import warnings

from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolops import RDKFingerprint, RenumberAtoms
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, \
    GetHashedTopologicalTorsionFingerprintAsBitVect, Properties, GetMACCSKeysFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.QED import properties, qed
from rdkit.Avalon.pyAvalonTools import GetAvalonFP, GetAvalonCountFP
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.base import TransformerMixin
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
        new_order = sorted([(y,x) for x,y in enumerate(new_order)])
        mol = RenumberAtoms(mol,[y for (x,y) in new_order]) 
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol


def smiles_to_mols(mols, ordered=True):
    """Convert a list of molecules to Chem.Mol"""
    res = []
    for mol in mols:
        res.append(to_mol(mol, ordered=ordered))
    return res


def explicit_bit_vect_to_array(bitvector):
    r"""
    Convert a bit vector into an array

    Arguments
    ----------
        bitvector: `rdkit.DataStructs.cDataStructs`
            The struct of interest as a bitvector

    Returns
    -------
        res: `numpy.ndarray`
            array of binary elements
    """
    return np.array(list(map(int, bitvector.ToBitString())))


def augmented_mol_properties(mol):
    r"""
    Get the set of molecular properties defined in rdkit for all molecule,
    in addition of the following props:

    * the number of heavy_atoms,
    * the number of structural alerts
    * qed value
    * boolean conclusion of Lipinski rules of 5
    * boolean conclusion of Veber rule
    * boolean conclusion of Ghose rule

    Arguments
    ----------
        mol: rdkit.Chem.Molecule
            the molecule of interest

    Returns
    -------
        props: list(float)
            All the props of interest
    """
    # we first need to compute the chirality for the stereo
    Chem.FindMolChiralCenters(mol, force=True)
    p = Properties()
    d = OrderedDict(zip(p.GetPropertyNames(),
                        p.ComputeProperties(mol)))
    d['heavy_atoms'] = mol.GetNumHeavyAtoms()

    # qed
    qed_props = properties(mol)
    d['ALERTS'] = qed_props.ALERTS
    d['qed'] = qed(mol)

    # Lipinski rule
    d['Lipinski'] = float(d['exactmw'] < 500 and d['lipinskiHBD'] < 5 and
                          d['lipinskiHBA'] < 10 and d['CrippenClogP'] < 5)

    # Veber rule
    d['Veber'] = float(d['NumRotatableBonds'] < 500 and d['tpsa'] <= 140)

    # Ghose rule
    d['Ghose'] = float(d['tpsa'] <= 140 and -0.4 < d['CrippenClogP'] < 5.6 and
                       160 <= d['exactmw'] < 480 and 20 <= d['heavy_atoms'] <= 70)
    return list(d.values())


def get_bond_feats(bond, add_bond=False):
    # Encode bond type as a feature vector
    if not add_bond:
        return np.ones(1, dtype=int)
    bond_type = bond.GetBondType()
    encoding = np.zeros(len(const.BOND_TYPES), dtype=int)
    for i, v in enumerate(const.BOND_TYPES):
        if v == bond_type:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found in accepted list
        encoding[0] = 1
    return encoding


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


class MoleculeTransformer(TransformerMixin):
    r"""
    Transform a molecule (rdkit.Chem.Mol object) into a feature representation.
    This class is an abstract class, and all its children are expected to implement the `_transform` method.
    """

    def __init__(self):
        super(MoleculeTransformer, self).__init__()

    def fit(self, X, y=None, **fit_params):
        return self

    def _transform(self, mol):
        r"""
        Compute features for a single molecule.
        This method need to be implemented by each child that inherits from MoleculeTransformer
        :raises NotImplementedError: if the method is not implemented by the child class
        Arguments
        ----------
            mol: Chem.Mol
                molecule to transform into features

        Returns
        -------
            features: the list of features

        """
        raise NotImplementedError('Missing implementation of _transform.')

    def transform(self, mols, ignore_errors=True, **kwargs):
        r"""
        Compute the features for a set of molecules.

        .. note::
            Note that depending on the `ignore_errors` argument, all failed
            featurization (caused whether by invalid smiles or error during
            data transformation) will be substitued by None features for the
            corresponding molecule. This is done, so you can find the positions
            of these molecules and filter them out according to your own logic.

        Arguments
        ----------
            mols: list(Chem.Mol) or list(str)
                a list containing smiles or Chem.Mol objects
            ignore_errors: bool, optional
                Whether to silently ignore errors
            kwargs:
                named arguments that are to be passed to the `to_mol` function.

        Returns
        --------
            features: a list of features for each molecule in the input set
        """

        features = []
        for i, mol in enumerate(mols):
            feat = None
            if ignore_errors:
                try:
                    mol = to_mol(mol, **kwargs)
                    feat = self._transform(mol)
                except:
                    pass
            else:
                mol = to_mol(mol, **kwargs)
                feat = self._transform(mol)
            features.append(feat)
        return features

    def __call__(self, mols, ignore_errors=True, **kwargs):
        r"""
        Calculate features for molecules. Using __call__, instead of transform. This function
        will force ignore_errors to be true, regardless of your original settings, and is offered
        mainly as a shortcut for data preprocessing. Note that most Transfomers allow you to specify
        a return datatype.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                SMILES of the molecules to be transformed
            ignore_errors: bool, optional
                Whether to ignore errors and silently fallback
                (Default value = True)
            kwargs: Named parameters for the transform method

        Returns
        -------
            feats: array
                list of valid features
            ids: array
                all valid molecule positions that did not failed during featurization
        """
        feats = self.transform(mols, ignore_errors=ignore_errors, **kwargs)
        ids = []
        for f_id, feat in enumerate(feats):
            if feat is not None:
                ids.append(f_id)
        return list(filter(None.__ne__, feats)), ids


class FingerprintsTransformer(MoleculeTransformer):
    r"""
    Fingerprint molecule transformer.
    This transformer is able to compute various fingerprints regularly used in QSAR modeling.

    Arguments
    ----------
        kind: str, optional
            Name of the fingerprinting method used. Should be one of
            {'global_properties', 'atom_pair', 'topological_torsion',
            'morgan_circular', 'estate', 'avalon_bit', 'avalon_count', 'erg',
            'rdkit', 'maccs'}
            (Default value = 'morgan_circular')
        length: int, optional
            Length of the fingerprint to use
            (Default value = 2000)

    Attributes
    ----------
        kind: str
            Name of the fingerprinting technique used
        length: int
            Length of the fingerprint to use
        fpfun: function
            function to call to compute the fingerprint
    """
    MAPPING = OrderedDict(
        global_properties=lambda x, params: augmented_mol_properties(x),
        # physiochemical=lambda x: GetBPFingerprint(x),
        atom_pair=lambda x, params: GetHashedAtomPairFingerprintAsBitVect(
            x, **params),
        topological_torsion=lambda x, params: GetHashedTopologicalTorsionFingerprintAsBitVect(
            x, **params),
        morgan_circular=lambda x, params: GetMorganFingerprintAsBitVect(
            x, 2, **params),
        estate=lambda x, params: FingerprintMol(x)[0],
        avalon_bit=lambda x, params: GetAvalonFP(x, **params),
        avalon_count=lambda x, params: GetAvalonCountFP(x, **params),
        erg=lambda x, params: GetErGFingerprint(x),
        rdkit=lambda x, params: RDKFingerprint(x, **params),
        maccs=lambda x, params: GetMACCSKeysFingerprint(x)
    )

    def __init__(self, kind='morgan_circular', length=2000):
        super(FingerprintsTransformer, self).__init__()
        if not (isinstance(kind, str) and (kind in FingerprintsTransformer.MAPPING.keys())):
            raise ValueError("Argument kind must be in: " +
                             ', '.join(FingerprintsTransformer.MAPPING.keys()))
        self.kind = kind
        self.length = length
        self.fpfun = self.MAPPING.get(kind, None)
        if not self.fpfun:
            raise ValueError("Fingerprint {} is not offered".format(kind))
        self._params = {}
        self._params.update(
            {('fpSize' if kind == 'rdkit' else 'nBits'): length})

    def _transform(self, mol):
        r"""
        Transforms a molecule into a fingerprint vector
        :raises ValueError: when the input molecule is None

        Arguments
        ----------
            mol: rdkit.Chem.Mol
                Molecule of interest

        Returns
        -------
            fp: np.ndarray
                The computed fingerprint

        """

        if mol is None:
            raise ValueError("Expecting a Chem.Mol object, got None")
        # expect cryptic rdkit errors here if this fails, #rdkitdev
        fp = self.fpfun(mol, self._params)
        if isinstance(fp, ExplicitBitVect):
            fp = explicit_bit_vect_to_array(fp)
        else:
            fp = list(fp)
        return fp

    def transform(self, mols, **kwargs):
        r"""
        Transforms a batch of molecules into fingerprint vectors.

        .. note::
            The recommended way is to use the object as a callable.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                List of SMILES or molecules
            kwargs: named parameters for transform (see below)

        Returns
        -------
            fp: array
                computed fingerprints of size NxD, where D is the
                requested length of features and N is the number of input
                molecules that have been successfully featurized.

        """
        return super(FingerprintsTransformer, self).transform(mols, **kwargs)

    def __call__(self, mols, dtype=torch.long, cuda=False, **kwargs):
        r"""
        Transforms a batch of molecules into fingerprint vectors,
        and return the transformation in the desired data type format as well as
        the set of valid indexes.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                The list of input smiles or molecules
            dtype: torch.dtype or numpy.dtype, optional
                Datatype of the transformed variable.
                Expect a tensor if you provide a torch dtype, a numpy array if you provide a
                numpy dtype (supports valid strings) or a vanilla int/float. Any other option will
                return the output of the transform function.
                (Default value = torch.long)
            cuda: bool, optional
                Whether to transfer tensor on the GPU (if output is a tensor)
            kwargs: named parameters for transform (see below)

        Returns
        -------
            fp: array
                computed fingerprints (in `dtype` datatype) of size NxD,
                where D is the requested length of features and N is the number
                of input molecules that have been successfully featurized.
            ids: array
                all valid molecule positions that did not failed during featurization

        """
        fp, ids = super(FingerprintsTransformer, self).__call__(mols, **kwargs)
        if is_numpy(dtype):
            fp = np.array(fp, dtype=dtype)
        elif is_tensor(dtype):
            fp = to_tensor(fp, gpu=cuda, dtype=dtype)
        else:
            raise(TypeError('The type {} is not supported'.format(dtype)))
        return fp, ids

class GraphTransformer(MoleculeTransformer):
    def __init__(self, mol_size=[0, 100], explicit_H=False, all_feat=True, add_bond=False):
        # if this is not set, packing of graph would be expected later
        self.mol_size = mol_size
        self.n_atom_feat = 0
        self.explicit_H = explicit_H
        self.use_chirality = True
        self.all_feat = all_feat
        self.add_bond = add_bond
        self._set_num_features()
        self.bond_dim = 1
        if add_bond:
            self.bond_dim = len(const.BOND_TYPES)

    @staticmethod
    def atom_dim():
        return len(const.ATOM_LIST) + 1

    def _set_num_features(self):
        r"""Compute the number of features for each atom and bond
        """
        self.n_atom_feat = 0
        # add atom type required
        self.n_atom_feat += len(const.ATOM_LIST) + 1
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
                if (self.mol_size[0] and self.mol_size[0] > num_atom) or (self.mol_size[1] and num_atom > self.mol_size[1]):
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
            atom_feats = one_of_k_encoding(atom.GetSymbol(), const.ATOM_LIST)
            if self.all_feat:
                atom_add_feat_arrays.append(
                    get_add_feats(atom, explicit_H=self.explicit_H, use_chirality=self.use_chirality))
            atom_arrays.append(atom_feats)
            for n_pos, neighbor in enumerate(atom.GetNeighbors()):
                n_idx = neighbor.GetIdx()
                # do not exceed hard limit on the maximum number of atoms
                # allowed
                bond = mol.GetBondBetweenAtoms(a_idx, n_idx)
                bond_feat = get_bond_feats(bond, self.add_bond)
                if n_idx < n_atoms:
                    adj_matrix[n_idx, a_idx] = bond_feat
                    adj_matrix[a_idx, n_idx] = bond_feat

                # keep self loop to empty, then rewrite graph conv

        n_atom_shape = len(atom_arrays[0])
        atom_matrix = np.zeros(
            (n_atoms, n_atom_shape)).astype(np.int)
        for idx, atom_array in enumerate(atom_arrays):
            atom_matrix[idx, :] = atom_array
        if atom_add_feat_arrays and self.all_feat:
            feat_matrix = np.concatenate([atom_matrix, np.asarray(atom_add_feat_arrays)], axis=1)
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