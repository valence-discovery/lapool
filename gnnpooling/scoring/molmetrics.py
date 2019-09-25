import gzip
import math
import numpy as np
import os
from functools import partial
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import Draw
from rdkit import DataStructs
from sklearn.base import BaseEstimator
from torch import nn
from joblib import Parallel, delayed
try:
   import cPickle as pickle
except ImportError:
   import pickle

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

_sas_model = None

######## Some Helper function ###########

def _read_fragments():
    global _sas_model
    name = os.path.join(os.path.dirname(__file__), "fpscores.pkl.gz")
    _sas_model = pickle.load(gzip.open(name))
    outDict = {}
    for i in _sas_model:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _sas_model = outDict


def _fail_silently(fn, arg):
    try:
        return fn(arg)
    except:
        return None


def _set_default(x, val=0):
    return val if x is None else x


def _remap(x, x_min, x_max):
    return (np.asarray(x) - x_min) / (x_max - x_min)


def _compute_diversity(mol, fps):
    ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(
        ref_fps, fps, returnDistance=True)
    score = np.mean(dist)
    return score


def _compute_SAS(mol):
    # peter ertl & greg landrum, september 2013
    # TODO: add Novartis copyright before release
    
    if mol is None:
        return 10

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if _sas_model is None:
        _read_fragments()

    fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _sas_model.get(sfp, -4) * v

    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


class Tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    def __init__(self, query_structure):
        if isinstance(query_structure, str):
            query_structure = Chem.MolFromSmiles(query_structure)
        self.query_structure = query_structure
        self.query_fp = AllChem.GetMorganFingerprint(
            self.query_structure, 2, useCounts=True, useFeatures=True)

    def __call__(self, mol, query=None):
        if not query:
            query = self.query_fp
        else:
            query = AllChem.GetMorganFingerprint(
                query, 2, useCounts=True, useFeatures=True)
        fp = AllChem.GetMorganFingerprint(
            mol, 2, useCounts=True, useFeatures=True)
        score = DataStructs.TanimotoSimilarity(query, fp)
        return float(score)


class ActivityModel:
    """Scores based on some model for activity.
    We do not expect the model weight to be updated here
    As we assume the model is already perfect :)
    """
    def __init__(self, model, tasks=None, ref_score=1.0):
        self.clf = model
        self.ref_score = ref_score
        self.tasks = tasks

    def __call__(self, mols):
        if isinstance(self.clf, nn.Module):
            self.clf.eval()
            with torch.no_grad():
                score = self.clf(mols)
        elif isinstance(self.clf, BaseEstimator):
            if not isinstance(mols, np.ndarray):
                mols = np.asarray(mols)
            if len(mols.shape) == 1:
                mols = mols.reshape(1, -1)
            score = self.clf.predict(mols)
        else:
            raise ValueError("Model is not valid")
        if self.tasks is not None:
            score = score[:, self.tasks]
        if len(score.shape)> 1:
            score = score.mean(-1)
        return (score - ref_score) / ref_score


class MolMetrics(object):

    @staticmethod
    def uniqueness(mols):
        generated_sm = list(map(lambda x: Chem.MolToSmiles(x), mols))
        return len(set(generated_sm)) / len(generated_sm)

    @staticmethod
    def novelty(mols, smiles):
        novels = []
        for mol in mols:
            sm = Chem.MolToSmiles(mol)
            if sm not in smiles:
                novels.append(sm)
        return float(len(novels))/len(mols)

    @staticmethod
    def sssr_metric(mols):
        overlapped_molecule = 0
        for mol in mols:
            ssr = Chem.GetSymmSSSR(mol)
            overlap_flag = False
            for idx1 in range(len(ssr)):
                for idx2 in range(idx1+1, len(ssr)):
                    if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                        overlap_flag = True
            if overlap_flag:
                overlapped_molecule += 1
        return overlapped_molecule/len(mols)

    @staticmethod
    def qed(mols, norm=False):
        return np.array(list(map(_set_default, [_fail_silently(QED.qed, mol) for mol in mols])))

    @staticmethod
    def partition_coefficient(mols, norm=False):
        """Water-octanol partition coefficient or logP"""
        scores = [_set_default(_fail_silently(
            Crippen.MolLogP, mol), -3) for mol in mols]
        scores = np.clip(_remap(scores, -2.12178879609,
                                6.0429063424), 0.0, 1.0) if norm else scores
        return scores

    @staticmethod
    def sas(mols, norm=False):
        """Return something in range 1-10"""
        scores =  Parallel(n_jobs=2)(delayed(_compute_SAS)(mol) for mol in mols)
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(_remap(scores, 8, 1), 0.0, 1.0) if norm else scores
        return scores

    @staticmethod
    def diversity(mols, refs, max_choice=None, norm=False):
        if max_choice is None:
            rand_mols = refs
        rand_mols = np.random.choice(refs, max_choice)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048) for mol in rand_mols]
        scores = np.array(
            [_set_default(_fail_silently(_compute_diversity(x, fps), 0)) for x in mols])
        scores = np.clip(_remap(scores, 0.9, 0.945),
                         0.0, 1.0) if norm else scores
        return scores

    @staticmethod
    def validity(mols):
        return 1- np.array([(mol is None) for mol in mols])

    @staticmethod
    def wrong_atoms(mols, unwanted_atoms=["*"]):
        return 1 - np.array([(np.mean([(atom.GetSymbol() in unwanted_atoms) for atom in mol.GetAtoms()]) if mol else 1) for mol in mols])



def all_scores(mols, data, norm=False):
    """
    :params: mol is the generated molecules in Chem.Mol
    :data: is all the smiles in the dataset
    """
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'QED score': MolMetrics.qed(mols),
        'logP score': MolMetrics.partition_coefficient(mols, norm=norm),
        'SA score': MolMetrics.sas(mols, norm=norm),
        'diversity score': MolMetrics.diversity(mols, data)}.items()}

    m1 = { 'unique score': MolMetrics.uniqueness(mols) * 100,
          'novel score': MolMetrics.novelty(mols, data) * 100,
          'sssr_metric': MolMetrics.sssr_metric(mols),
          'validity': MolMetrics.validity(mols)*100,
          "existence":MolMetrics.wrong_atoms(mols)*100}

    return m0, m1