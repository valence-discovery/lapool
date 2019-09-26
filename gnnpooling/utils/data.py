import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import sys
import torch

from functools import partial
from rdkit import Chem
from rdkit.Chem.rdmolops import RenumberAtoms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from gnnpooling.utils import const
from gnnpooling.utils.tensor_utils import to_tensor, is_tensor, one_of_k_encoding, is_numpy
from gnnpooling.utils.graph_utils import pad_graph, pad_feats
from gnnpooling.utils.transformers import GraphTransformer, FingerprintsTransformer, to_mol, smiles_to_mols

# from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
MAX_N = None

BENCHMARKS = {"ENZYMES", "DD", "FRANKENSTEIN", "PROTEINS"}


# Function taken from https://github.com/RexYing/diffpool
def read_graphfile(dataname, datadir="data", max_nodes=None, min_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset 
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        if (max_nodes is not None and G.number_of_nodes() > max_nodes) or (
                min_nodes is not None and G.number_of_nodes() < min_nodes):
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs


def read_dense_graph_data(benchmark, min_num_nodes=0, max_num_nodes=None):
    if benchmark not in BENCHMARKS:
        raise ValueError("Check benchmark values.")
    graphs = read_graphfile(dataname=benchmark, min_nodes=min_num_nodes, max_nodes=max_num_nodes)
    labels = []
    for G in graphs:
        for u in G.nodes():
            if G.node[u].get("feat") is None:
                # fall back to node label if node attributes are not found
                G.node[u]['feat'] = np.array(G.node[u]['label'])
        labels.append(G.graph['label'])

    return graphs, labels


class NetworkXGraphDataset(Dataset):
    def __init__(self, graphs, targets, w=None, device='cpu'):
        self.w = w
        self.device = device
        self.y = np.array(targets)

        # Extract Adjacency and feature matrices from Networkx graph objects
        self.A = np.array([np.expand_dims(nx.to_numpy_matrix(G, dtype=np.float32), 2) for G in graphs])
        X = []
        for G in graphs:
            feat_matrix = []
            for node in G.nodes():
                feat_matrix.append(G.node[node]['feat'])
            X.append(np.array(feat_matrix))
        self.X = np.array(X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        adj = torch.Tensor(self.A[idx]).to(self.device)
        feat = torch.Tensor(self.X[idx]).to(self.device)
        target = torch.Tensor([self.y[idx]]).float().to(self.device)

        if self.w is not None:
            w = torch.Tensor(self.w[idx]).float().to(self.device)
            return (adj, feat), target, w
        else:
            return (adj, feat), target


class GraphToFingerprintDataset(Dataset):
    def __init__(self, smiles_mols, targets=None, w=None, device='cpu', with_bond=False, kind="maccs", length=None,
                 min_size=0):
        """
        Specialized dataset for supervised molecular graph to fingerprint prediction experiment,
        where the molecular graph is represented as an adjacency matrix, feature encoded representation
        (as returned by Adjacency graph dataset).

        Arguments
        --------------
        inputs: np.array
                Array of smiles molecule representations.
        """

        self.w = w
        X, ids_x = GraphTransformer(add_bond=with_bond, all_feat=True, mol_size=[min_size, 100])(smiles_mols,
                                                                                                 dtype=np.float32)
        self.raw_mols = smiles_to_mols(smiles_mols[ids_x])
        if targets is None:
            y, ids_y = FingerprintsTransformer(kind=kind, length=length)(np.asarray(smiles_mols)[ids_x], dtype=np.int32)
        else:
            y = targets[ids_x].astype(np.int32)
        self.device = device

        # Extract adjacency matrix and graph features
        A, X = zip(*X)
        A = np.asarray(A)
        X = np.asarray(X)

        # Keep only examples where both transformations succeeded
        if targets is None:
            self.X, self.A, self.y = X[ids_y], A[ids_y], y
            if w is not None:
                self.w = self.w[ids_x][ids_y]
        else:
            self.X, self.A, self.y = X, A, y
            if w is not None:
                self.w = self.w[ids_x]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        adj = torch.Tensor(self.A[idx]).to(self.device)
        feat = torch.Tensor(self.X[idx]).to(self.device)
        target = torch.Tensor(self.y[idx]).float().to(self.device)

        if self.w is not None:
            w = torch.Tensor(self.w[idx]).float().to(self.device)
            return (adj, feat), target, w
        else:
            return (adj, feat), target


class GraphDataset(Dataset):
    def __init__(self, X, y, mols, w=None, cuda=False, pad_to=-1, **kwargs):
        self.cuda = cuda
        self.w = None
        self.G = []
        self.feat = []
        self.add_feat = []
        self.mols = mols
        self.pad = partial(pad_graph, max_num_node=pad_to)
        fake_atom = to_tensor(one_of_k_encoding('*', const.ATOM_LIST), dtype=torch.float32, gpu=cuda)
        self.pad_x = partial(pad_feats, no_atom_tensor=fake_atom, max_num_node=pad_to)
        if len(X) > 0:
            self.G, self.feat, *self.add_feat = zip(*X)
            self.G = list(self.G)
            self.feat = list(self.feat)
            self.y = to_tensor(y, gpu=self.cuda, dtype=torch.float32)
            if self.add_feat:
                self.add_feat = self.add_feat[0]
            if w is not None:
                self.w = w.reshape(y.shape[0], -1)
                self.w = to_tensor(self.w, gpu=self.cuda, dtype=torch.float32)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.G)

    @property
    def X(self):
        G, F = self.G, self.feat
        G = [self.pad(to_tensor(g_i, gpu=self.cuda, dtype=torch.float32)) for g_i in G]
        F = [self.pad_x(to_tensor(f_i, gpu=self.cuda, dtype=torch.float32)) for f_i in F]
        return list(zip(G, F))

    def __getitem__(self, idx):
        g_i, f_i = self.G[idx], self.feat[idx]
        if not isinstance(g_i, torch.Tensor):
            g_i = self.pad(to_tensor(g_i, gpu=self.cuda, dtype=torch.float32))
        if not isinstance(f_i, torch.Tensor):
            f_i = self.pad_x(to_tensor(f_i, gpu=self.cuda, dtype=torch.float32))
        X_i = (g_i, f_i)
        if self.add_feat:
            af_i = self.add_feat[idx]
            if not isinstance(af_i, torch.Tensor):
                af_i = self.pad_x(to_tensor(af_i, gpu=self.cuda, dtype=torch.float32), no_atom_tensor=None)
            X_i += (af_i,)
        y_i = self.y[idx, None]
        if self.w is not None:
            w_i = self.w[idx, None]
            return (*X_i, self.mols[idx]), y_i, w_i
        return (*X_i, self.mols[idx]), y_i


def batch_generator_sup(dataset, batch_size=32, infinite=True, shuffle=True, balance=False, with_mols=False):
    # If the batch_size is chosen to be -1, then we assume all the elements are in the same batch
    if batch_size == -1:
        while True:
            yield dataset  # return the complete dataset
    else:
        # Prepare the indexes necessary for the batch selection
        loop = 0
        idx = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(idx)

        # Loop the dataset once (for testing and validation) or infinite times (for training)
        while loop < 1 or infinite:

            # Loop the whole dataset and create the batches according to batch_size
            for i in range(0, len(dataset), batch_size):
                start = i  # starting index of the batch
                end = min(i + batch_size, len(dataset))  # ending index of the batch
                a = [dataset[idx[ii]] for ii in range(start, end)]  # Generate the batch 'a'
                if balance:
                    x, y, w = zip(*a)
                else:
                    x, y = zip(*a)
                adj, feat = zip(*x)
                X = (adj, feat)
                if with_mols:
                    mols = [dataset.raw_mols[idx[ii]] for ii in range(start, end)]
                    X = (adj, feat, mols)
                y = torch.cat([y.unsqueeze(0) for y in y], dim=0)
                if balance:
                    w = torch.cat([w.unsqueeze(0) for w in w], dim=0)
                    yield X, y, w
                else:
                    yield X, y  # return the x and y values of the batch
            loop += 1


def batch_generator(dataset, batch_size=32, infinite=True, shuffle=True, split=0):
    # If the batch_size is chosen to be -1, then we assume all the elements are in the same batch
    if batch_size == -1:
        while True:
            yield dataset  # return the complete dataset
    else:
        # Prepare the indexes necessary for the batch selection
        loop = 0
        idx = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(idx)

        # Loop the dataset once (for testing and validation) or infinite times (for training)
        while loop < 1 or infinite:

            # Loop the whole dataset and create the batches according to batch_size
            for i in range(0, len(dataset), batch_size):
                start = i  # starting index of the batch
                # ending index of the batch
                end = min(i + batch_size, len(dataset))
                a = [dataset[idx[ii]]
                     for ii in range(start, end)]  # Generate the batch 'a'
                x, *y = zip(*a)
                y = [torch.cat(yy, dim=0) for yy in y]
                # if split:
                #     x = (x[0], *x[1].split([split, x[1].size(dim=-1)-split], dim=-1))
                x = zip(*x)
                # return the x and y values of the batch
                yield (x, *y)
            loop += 1


def graph_collate(batch):
    x, *y = zip(*batch)
    y = [torch.cat(yy, dim=0) for yy in y]
    x, mols = x
    return ((*x, mols), *y)


def standardize_data(file_name, smiles_column, save_path, delim_whitespace=False):
    """
    Utility function to shift the axis containing the smiles towards the first column of data.
    Used to produce the datasets found in standard_data (only applied to those with smiles out of place).

    """
    datafile = pd.read_csv(file_name, delim_whitespace=delim_whitespace)
    datafile.fillna(0, inplace=True)  # Add a 0 where data is missing
    data = datafile.values
    num_columns = data.shape[1]
    shift = num_columns - smiles_column
    data = np.roll(data, shift=shift, axis=1)
    # Strip string columns
    for col in range(1, data.shape[1]):
        if isinstance(data[0, col], str):
            data = np.delete(data, col, 1)
            col -= 1
    pd.DataFrame(data).to_csv(save_path, header=None, index=None, )


def read_data(file_name, num_column=-1, format='csv', has_header=True, max_n=-1):
    if format == 'csv':
        # Read the data file without the header
        if has_header:
            try:
                datafile = pd.read_csv(file_name, header=0, engine='python')
            except:
                datafile = pd.read_csv(file_name, header=0)
        else:
            try:
                datafile = pd.read_csv(file_name, header=None, engine='python')
            except:
                datafile = pd.read_csv(file_name, header=None)

        datafile.fillna(0, inplace=True)  # Add a 0 where data is missing
        data = datafile.values
        smiles = data[:, 0]  # The first column is the smiles
        if num_column == -1:
            y = data[:, 1:]
        else:
            y = data[:, -num_column:]  # Take the last n_col targets
    elif format == 'pickle':
        smiles = pd.read_pickle(file_name)
        smiles = list(filter(None, smiles))
        y = None
    else:
        raise ValueError('File format specified incorrectly.')
    return smiles[:max_n], y[:max_n]


def transform_data(smiles, y, min_size=0, max_size=50, all_feat=False, add_bond=False):
    # Initialize the transformer
    trans = GraphTransformer(mol_size=[min_size, max_size], all_feat=all_feat, add_bond=add_bond)
    # Call the transformer on the smiles using the __call__ method.
    X, ids = trans(smiles, dtype=np.float32, ignore_errors=False)
    # Keep only the ids where the transformation succeeded
    # (the failed transformations are not present in ids)
    y = y[ids, :].astype(np.float32)
    # Keep only the ids with more than min atoms
    raw_mols = smiles[ids]
    return X, y, smiles_to_mols(raw_mols)


def as_dataset(X, y, mols, cuda=True, **kwargs):
    dt = GraphDataset(X, y, mols, cuda=cuda, **kwargs)
    return dt


def load_mol_dataset(X, y, test_size=0.2, valid_size=0.2, device='cpu', shuffle=True, with_bond=False, balance=False,
                     stratify=False, mol_filter=0, type='sparse', **kwargs):
    if balance:
        wbalance = WeightBalancing(X, y)
        X, y, w = wbalance.transform(X, y)

        if stratify:
            strat = IterativeStratification(n_splits=2, order=2,
                                            sample_distribution_per_fold=[test_size, 1.0 - test_size])
            train_ix, test_ix = next(strat.split(X, y.astype(np.int32)))
            smiles_train, smiles_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix, :], y[test_ix, :]
            w_train, w_test = w[train_ix, :], w[test_ix, :]

            strat = IterativeStratification(n_splits=2, order=2,
                                            sample_distribution_per_fold=[valid_size, 1.0 - valid_size])
            train_ix, valid_ix = next(strat.split(smiles_train, y_train.astype(np.int32)))
            smiles_train, smiles_valid = smiles_train[train_ix], smiles_train[valid_ix]
            y_train, y_valid = y_train[train_ix, :], y_train[valid_ix, :]
            w_train, w_valid = w_train[train_ix, :], w_train[valid_ix, :]

        else:

            smiles_train, smiles_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w,
                                                                                           test_size=test_size,
                                                                                           shuffle=shuffle)

            smiles_train, smiles_valid, y_train, y_valid, w_train, w_valid = train_test_split(smiles_train, y_train,
                                                                                              w_train,
                                                                                              test_size=valid_size,
                                                                                              shuffle=shuffle)
    else:
        w_train = w_test = w_valid = None
        if stratify:
            strat = IterativeStratification(n_splits=2, order=2,
                                            sample_distribution_per_fold=[test_size, 1.0 - test_size])
            train_ix, test_ix = next(strat.split(X, y.astype(np.int32)))
            smiles_train, smiles_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix, :], y[test_ix, :]
            strat = IterativeStratification(n_splits=2, order=2,
                                            sample_distribution_per_fold=[valid_size, 1.0 - valid_size])
            train_ix, valid_ix = next(strat.split(smiles_train, y_train.astype(np.int32)))
            smiles_train, smiles_valid = smiles_train[train_ix], smiles_train[valid_ix]
            y_train, y_valid = y_train[train_ix, :], y_train[valid_ix, :]
        else:

            smiles_train, smiles_test, y_train, y_test = train_test_split(X, y,
                                                                          test_size=test_size,
                                                                          shuffle=shuffle)

            smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles_train, y_train,
                                                                            test_size=valid_size,
                                                                            shuffle=shuffle)
    if type == 'sparse':
        train_dt = GraphToFingerprintDataset(smiles_train, y_train, w=w_train, with_bond=with_bond, device=device,
                                             min_size=mol_filter)
        valid_dt = GraphToFingerprintDataset(smiles_valid, y_valid, w=w_valid, with_bond=with_bond, device=device,
                                             min_size=mol_filter)
        test_dt = GraphToFingerprintDataset(smiles_test, y_test, w=w_test, with_bond=with_bond, device=device,
                                            min_size=mol_filter)
    else:
        train_dt = NetworkXGraphDataset(smiles_train, y_train, w=w_train, device=device)

        valid_dt = NetworkXGraphDataset(smiles_valid, y_valid, w=w_valid, device=device)

        test_dt = NetworkXGraphDataset(smiles_test, y_test, w=w_test, device=device)

    return train_dt, valid_dt, test_dt


def load_gen_dataset(X, y, mols, test_size=0.2, valid_size=0.25, shuffle=True, balance=False, **kwargs):
    if balance:
        wbalance = WeightBalancing(X, y)
        X, y, w = wbalance.transform(X, y)
        x_train, x_test, y_train, y_test, w_train, w_test, mols_train, mols_test = train_test_split(X, y, w, mols,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)
        if valid_size:
            x_train, x_valid, y_train, y_valid, w_train, w_valid, mols_train, mols_valid = train_test_split(x_train,
                                                                                                            y_train,
                                                                                                            w_train,
                                                                                                            mols_train,
                                                                                                            test_size=valid_size,
                                                                                                            shuffle=shuffle)
    else:
        w_train = w_test = w_valid = None
        x_train, x_test, y_train, y_test, mols_train, mols_test = train_test_split(X, y, mols, test_size=test_size,
                                                                                   shuffle=shuffle)
        if valid_size:
            x_train, x_valid, y_train, y_valid, mols_train, mols_valid = train_test_split(x_train, y_train, mols_train,
                                                                                          test_size=valid_size,
                                                                                          shuffle=shuffle)

    train_dt = as_dataset(x_train, y_train, w=w_train, mols=mols_train, **kwargs)
    valid_dt = None if not valid_size else as_dataset(x_valid, y_valid, w=w_valid, mols=mols_valid, **kwargs)
    test_dt = as_dataset(x_test, y_test, w=w_test, mols=mols_test, **kwargs)

    return train_dt, valid_dt, test_dt


class WeightBalancing:

    def __init__(self, X, y, w=None):
        r"""Initializes transformation based on dataset statistics."""
        # Ensure dataset is binary
        if isinstance(y, list):
            y = np.asarray(y)
        if w is not None and isinstance(w, list):
            w = np.asarray(w)
        np.testing.assert_allclose(sorted(np.unique(y)), np.array([0., 1.]))
        if w is None:
            w = np.ones_like(y, dtype=np.float)
        weights = []
        for ind in range(y.shape[-1]):
            task_w = w[:, ind]
            task_y = y[:, ind]
            # Remove labels with zero weights
            task_y = task_y[task_w != 0]
            num_positives = np.count_nonzero(task_y)
            num_negatives = len(task_y) - num_positives
            if num_positives > 0:
                pos_weight = float(num_negatives) / num_positives
            else:
                pos_weight = 1
            neg_weight = 1
            weights.append((neg_weight, pos_weight))
        self.weights = weights

    def transform(self, X, y, w=None):
        r"""
        Transforms all internally stored data in a set of (X, y, w) arrays.
        """
        if w is None:
            w = np.ones_like(y, dtype=np.float)
        if isinstance(y, list):
            y = np.asarray(y)
        w_balanced = np.zeros_like(w)
        for ind in range(y.shape[-1]):
            task_y = y[:, ind]
            task_w = w[:, ind]
            zero_indices = np.logical_and(task_y == 0, task_w != 0)
            one_indices = np.logical_and(task_y == 1, task_w != 0)
            w_balanced[zero_indices, ind] = self.weights[ind][0]
            w_balanced[one_indices, ind] = self.weights[ind][1]
        return (X, y, w_balanced)
