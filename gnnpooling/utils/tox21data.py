import logging
import os
import deepchem
import torch
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader, Dataset

from gnnpooling.utils import const
from gnnpooling.utils.graph_utils import pad_graph, pad_feats
from gnnpooling.utils.transformers import GraphTransformer
from gnnpooling.utils.tensor_utils import to_tensor

logging.getLogger("deepchem").setLevel(logging.WARNING)


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

def load_tox21(split='stratified', dataset_file=os.path.join(data_dir, "tox21/tox21.csv.gz"), **kwargs):
    """Load Tox21 datasets. Does not do train/test split"""
    # Featurize Tox21 dataset

    tox21_tasks = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    featurizer = deepchem.feat.RawFeaturizer()
    loader = deepchem.data.CSVLoader(tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer, verbose=False)
    dataset = loader.featurize(dataset_file)

    splitters = {
            'index': deepchem.splits.IndexSplitter(),
            'random': deepchem.splits.RandomSplitter(),
            'scaffold': deepchem.splits.ScaffoldSplitter(),
            'butina': deepchem.splits.ButinaSplitter(),
            'stratified': deepchem.splits.RandomStratifiedSplitter()
    }
    splitter = splitters[split]


    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train, valid, test = splitter.train_valid_test_split(
            dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=kwargs.get("seed"))

    all_dataset = (train, valid, test)
    transformers = [
            deepchem.trans.BalancingTransformer(transform_w=True, dataset=train)
    ]    
    for transformer in transformers:
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)

    return tox21_tasks, all_dataset, transformers


class MolDataset(Dataset):
    def __init__(self, X, y, mols, w=None, cuda=False, pad_to=-1, **kwargs):
        self.cuda = cuda
        self.adj = []
        self.x = []
        self.w = None
        self.mols = mols
        if pad_to is None:
            pad_to = -1

        l = 0 or X[0][-1].shape[-1]
        fake_atom = to_tensor(np.zeros(l), dtype=torch.float32, gpu=cuda)
        self.pad_x = partial(pad_feats, no_atom_tensor=fake_atom, max_num_node=pad_to)
        self.pad = partial(pad_graph, max_num_node=pad_to)
        
        if len(X) > 0:
            self.adj, self.x = zip(*X)
            self.adj = list(self.adj)
            self.x = list(self.x)
            self.y = to_tensor(y, gpu=self.cuda, dtype=torch.float32)
            if w is not None:
                self.w = w.reshape(y.shape[0], -1)
                self.w = to_tensor(self.w, gpu=self.cuda, dtype=torch.float32)
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.adj)

    @property
    def X(self):
        G, F = self.adj, self.x
        G = [self.pad(to_tensor(g_i, gpu=self.cuda, dtype=torch.float32)) for g_i in G]
        F = [self.pad_x(to_tensor(f_i, gpu=self.cuda, dtype=torch.float32)) for f_i in F]
        return list(zip(G, F))


    def __getitem__(self, idx):
        g_i, f_i = self.adj[idx], self.x[idx]
        true_nodes = g_i.shape[0]
        if not isinstance(g_i, torch.Tensor):
            g_i = self.pad(to_tensor(g_i, gpu=self.cuda, dtype=torch.float32)).squeeze() # remove edge dim if exist
        if not isinstance(f_i, torch.Tensor):
            f_i = self.pad_x(to_tensor(f_i, gpu=self.cuda, dtype=torch.float32))
        y_i = self.y[idx, None]
        # add mask for binary 
        m_i = torch.zeros(g_i.shape[-1])
        m_i[torch.arange(true_nodes)] = 1
        m_i = m_i.unsqueeze(-1)
        
        if self.w is not None:
            w_i = self.w[idx, None]
            return (g_i, f_i, m_i), self.mols[idx], y_i, w_i
        return (g_i, f_i, m_i), self.mols[idx], y_i


class MolDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, as_list=False, **kwargs):
        def graph_collate(batch):
            x, mols, *y = zip(*batch)            
            x = tuple(zip(*x))
            if not as_list:
                x = (torch.stack(x[0]), torch.stack(x[1]), torch.stack(x[2]))
            y = [torch.cat(yy, dim=0) for yy in y]
            return (x, mols, *y)

        super(MolDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=graph_collate, **kwargs)


def mol_graph_transformer(dataset, min_size=0, max_size=None, **kwargs):
    # Initialize the transformer
    # Call the transformer on the smiles using the __call__ method.
    trans = GraphTransformer(mol_size=[min_size, max_size], **kwargs)
    X, ids = trans(dataset.ids, dtype=np.float32, ignore_errors=False)
    # Keep only the ids where the transformation succeeded
    # (the failed transformations are not present in ids)
    y = dataset.y[ids, :]
    w = dataset.w[ids, :]
    # Keep only the ids with more than min atoms
    raw_mols = dataset.X[ids]
    return MolDataset(X, y, raw_mols, w=w, pad_to=max_size) 


def get_tox21(min_size=0, max_size=None, add_bond=False, seed=None):
    tox21_tasks, (train, valid, test), _ = load_tox21(seed=seed) 
    return as_dataset(train, valid, test, min_size=min_size, max_size=max_size, add_bond=add_bond)

def as_dataset(train, valid, test, min_size=0, max_size=None, add_bond=False, **kwargs):
    train_dt = mol_graph_transformer(train, add_bond=add_bond, min_size=min_size, max_size=max_size)
    valid_dt = mol_graph_transformer(valid, add_bond=add_bond, min_size=min_size, max_size=max_size)
    test_dt = mol_graph_transformer(test, add_bond=add_bond, min_size=min_size, max_size=max_size)
    in_size = train_dt.x[0].shape[-1]
    out_size = 1 if len(train_dt.y.shape) == 1 else train_dt.y.shape[-1]
    return train_dt, valid_dt, test_dt, MolDataLoader, in_size, out_size


def load_chembl(split='random', dataset_file="chembl_dataset_fragments.txt", **kwargs):
    """Load Tox21 datasets. Does not do train/test split"""
    # Featurize Tox21 dataset

    const.ATOM_LIST = ['O', 'C', 'F', 'Cl', 'Br', 'P', 'I', 'S', 'N']

    dataset_file = os.path.join(data_dir, dataset_file)
    file = pd.read_csv(dataset_file)
    tasks = list(file.head(0))[1:]

    featurizer = deepchem.feat.RawFeaturizer()
    loader = deepchem.data.CSVLoader(tasks=tasks, smiles_field="smiles", featurizer=featurizer, verbose=False)
    dataset = loader.featurize(dataset_file)

    splitters = {
            'index': deepchem.splits.IndexSplitter(),
            'random': deepchem.splits.RandomSplitter(),
            'scaffold': deepchem.splits.ScaffoldSplitter(),
            'butina': deepchem.splits.ButinaSplitter(),
            'stratified': deepchem.splits.RandomStratifiedSplitter()
    }
    splitter = splitters[split]


    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train, valid, test = splitter.train_valid_test_split(
            dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=kwargs.get("seed"))

    all_dataset = (train, valid, test)
    transformers = [
            deepchem.trans.BalancingTransformer(transform_w=True, dataset=train)
    ]    
    for transformer in transformers:
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)

    return tasks, all_dataset, transformers

