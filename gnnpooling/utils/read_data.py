import logging
import networkx as nx
import numpy as np
import os
import pandas as pd
import re

from deepchem.data.datasets import NumpyDataset
from deepchem.data import CSVLoader
from deepchem.feat import RawFeaturizer
from deepchem.splits import ButinaSplitter, IndexSplitter, RandomSplitter, RandomStratifiedSplitter, ScaffoldSplitter
from deepchem.trans import BalancingTransformer
from functools import partial
from sklearn.model_selection import train_test_split
from gnnpooling.utils.transformers import GraphTransformer, to_mol, smiles_to_mols
from gnnpooling.utils.datasets import NetworkXGraphDataset, MolDataset, GenMolDataset
from gnnpooling.utils import const

logging.getLogger("deepchem").setLevel(logging.ERROR)

DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../../data")
BENCHMARKS = {"ENZYMES", "DD", "FRANKENSTEIN",
              "PROTEINS", 'ALERTS', 'FRAGMENTS', 'TOX21'}

# Function taken from https://github.com/RexYing/diffpool


def _read_graphfile(dataname, datadir="data", max_nodes=None, min_nodes=None):
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
                attrs = [float(attr) for attr in re.split(
                    "[,\s]+", line) if not attr == '']
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


def _load_dense_dataset(dataset, valid_size=0.1, test_size=0.1, min_size=0, max_size=None, **kwargs):

    train_size = 1.0 - (test_size + valid_size)
    graphs = _read_graphfile(
        dataname=dataset, min_nodes=min_size, max_nodes=max_size)
    labels = []
    for G in graphs:
        for u in G.nodes():
            if G.node[u].get("feat") is None:
                # fall back to node label if node attributes are not found
                G.node[u]['feat'] = np.array(G.node[u]['label'])
        labels.append(G.graph['label'])
    n_tasks = len(set(labels))
    labels = np.asarray(labels)
    dataset = NumpyDataset(graphs, y=labels, n_tasks=n_tasks)
    splitter = RandomSplitter()
    # splits.RandomStratifiedSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset,
        frac_train=train_size,
        frac_valid=valid_size,
        frac_test=test_size)

    datasets = []
    for dt in (train, valid, test):
        datasets.append(NetworkXGraphDataset(
            dt.X, dt.y, w=None, pad_to=max_size))

    in_size = datasets[0].X[0].shape[-1]
    return datasets, in_size, n_tasks


def _load_mol_dataset(dataset_file, tasks, split="stratified", test_size=0.1, valid_size=0.1, min_size=0, max_size=None, **kwargs):

    train_size = 1.0 - (test_size + valid_size)
    featurizer = RawFeaturizer()
    loader = CSVLoader(tasks=tasks, smiles_field="smiles",
                       featurizer=featurizer, verbose=False, log_every_n=10000)
    dataset = loader.featurize(dataset_file)

    splitters = {
        'index': IndexSplitter(),
        'random': RandomSplitter(),
        'scaffold': ScaffoldSplitter(),
        'butina': ButinaSplitter(),
        'stratified': RandomStratifiedSplitter()
    }

    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(
        dataset,
        frac_train=train_size,
        frac_valid=valid_size,
        frac_test=test_size)

    # compute data balance information on train
    balancer = BalancingTransformer(transform_w=True, dataset=train)
    train = balancer.transform(train)
    valid = balancer.transform(valid)
    test = balancer.transform(test)
    transformer = GraphTransformer(mol_size=[min_size, max_size], **kwargs)
    datasets = []
    for dt in (train, valid, test):
        X, ids = transformer(dt.ids, dtype=np.float32, ignore_errors=False)
        y = dt.y[ids, :]
        w = dt.w[ids, :]
        raw_mols = dt.X[ids]
        datasets.append(MolDataset(X, y, raw_mols, w=w, pad_to=max_size))

    in_size = X[0][-1].shape[-1]
    out_size = 1 if len(y.shape) == 1 else y.shape[-1]
    return datasets, in_size, out_size


def load_supervised_dataset(dataset, min_size=0, max_size=None, test_size=0.1, valid_size=0.1, **kwargs):

    dataset = dataset.upper()

    if dataset not in BENCHMARKS:
        raise ValueError(
            f"Unknown dataset {dataset}, accepted values are {BENCHMARKS}")

    if dataset == 'TOX21':
        tasks = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        data_path = os.path.join(DATA_DIR, f'TOX21/tox21.csv.gz')
        atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B',
                     'K', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Cu', 'Ni', 'Cd', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg']
        return _load_mol_dataset(data_path, tasks, test_size=test_size, valid_size=valid_size, min_size=min_size, max_size=max_size, atom_list=atom_list, **kwargs)

    elif dataset in ['FRAGMENTS', 'ALERTS']:
        data_path = os.path.join(
            DATA_DIR, f'chembl_dataset_{dataset.lower()}.txt')
        f = pd.read_csv(data_path)
        tasks = list(f.head(0))[1:]
        atom_list = ['O', 'C', 'F', 'Cl', 'Br', 'P', 'I', 'S', 'N']
        return _load_mol_dataset(data_path, tasks, test_size=test_size, valid_size=valid_size, min_size=min_size, max_size=max_size, atom_list=atom_list, **kwargs)

    else:
        return _load_dense_dataset(dataset, test_size=test_size, valid_size=valid_size, min_size=min_size, max_size=max_size)


def read_gen_data(dataset, min_size=0, max_size=None, valid_size=0.15, test_size=0.15, max_n=-1,**kwargs):
    in_size = None
    atom_list = const.ATOM_LIST
    if dataset.lower() == 'qm9':
        in_size = 5
        atom_list = ['C', 'N', 'O', 'F']
        kwargs.update(atom_list=atom_list)
        data = pd.read_csv(os.path.join(DATA_DIR, 'qm9.csv'))
        max_size  =  max_size or 9
    else:
        data = pd.read_csv(dataset)
        atom_list = set([])
        old_max_size = max_size
        max_size = 0
        for m in data.values[:, 0]:    
            m = to_mol(m)
            if m:
                all_atoms = [a.GetSymbol() for a in m.GetAtoms()]
                atom_list.update(all_atoms)
                max_size = max(max_size, len(all_atoms))
        atom_list = list(sorted(atom_list))
        kwargs.update(atom_list=atom_list)
        in_size = len(atom_list) + 1
        if old_max_size:
            max_size = old_max_size
    
    if max_n >0:
        data = data.sample(max_n)

    x_train = data.values[:, 0]
    x_train, x_valid = train_test_split(x_train, test_size=valid_size, shuffle=True)
    x_train, x_test = train_test_split(x_train, test_size=test_size, shuffle=True)
    # all_feat=False, add_bond=True, atom_list = ['C', 'N', 'O', 'F'] 
    transformer = GraphTransformer(mol_size=[min_size, max_size], **kwargs)
    #def __init__(self, smiles, max_size, transformer, cuda=False, pad_to=-1, **kwargs):
    train_dt = GenMolDataset(x_train, transformer, pad_to=max_size)
    valid_dt = GenMolDataset(x_valid, transformer, pad_to=max_size)
    test_dt = GenMolDataset(x_test,  transformer, pad_to=max_size)
    return train_dt, test_dt, valid_dt, in_size, atom_list, max_size