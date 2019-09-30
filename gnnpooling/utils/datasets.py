import networkx as nx
import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from gnnpooling.utils.tensor_utils import to_tensor, is_tensor, one_of_k_encoding, is_numpy
from gnnpooling.utils.graph_utils import pad_graph, pad_feats


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
        self.pad_x = partial(
            pad_feats, no_atom_tensor=fake_atom, max_num_node=pad_to)
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
        G = [self.pad(to_tensor(g_i, gpu=self.cuda, dtype=torch.float32))
             for g_i in G]
        F = [self.pad_x(to_tensor(f_i, gpu=self.cuda, dtype=torch.float32))
             for f_i in F]
        return list(zip(G, F))

    def __getitem__(self, idx):
        g_i, f_i = self.adj[idx], self.x[idx]
        true_nodes = g_i.shape[0]
        if not isinstance(g_i, torch.Tensor):
            # remove edge dim if exist
            g_i = self.pad(to_tensor(g_i, gpu=self.cuda,
                                     dtype=torch.float32)).squeeze()
        if not isinstance(f_i, torch.Tensor):
            f_i = self.pad_x(
                to_tensor(f_i, gpu=self.cuda, dtype=torch.float32))
        y_i = self.y[idx, None]
        # add mask for binary
        m_i = torch.zeros(g_i.shape[-1])
        m_i[torch.arange(true_nodes)] = 1
        m_i = m_i.unsqueeze(-1)

        if self.w is not None:
            w_i = self.w[idx, None]
            return (g_i, f_i, m_i), self.mols[idx], y_i, w_i
        return (g_i, f_i, m_i), self.mols[idx], y_i


class GraphDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, as_list=False, **kwargs):
        def graph_collate(batch):
            x, mols, *y = zip(*batch)
            x = tuple(zip(*x))
            if not as_list:
                x = [torch.stack(x[i]) for i in range(len(x))]
            y = [torch.cat(yy, dim=0) for yy in y]
            return (tuple(x), mols, *y)

        super(GraphDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=graph_collate, **kwargs)


class NetworkXGraphDataset(Dataset):
    def __init__(self, graphs, targets, w=None, pad_to=None, device='cpu'):
        self.w = w
        self.device = device
        self.y = np.array(targets)
        self.pad_to = pad_to
        # Extract Adjacency and feature matrices from Networkx graph objects
        self.A = []
        self.X = []
        for G in graphs:
            a = np.expand_dims(nx.to_numpy_matrix(G, dtype=np.float32), 2)
            self.A.append(a)
            feat_matrix = []
            for node in G.nodes():
                feat_matrix.append(G.node[node]['feat'])
            self.X.append(np.array(feat_matrix))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        true_nodes = self.A[idx].shape[0]
        adj = pad_graph(torch.Tensor(self.A[idx]).to(
            self.device), max_num_node=self.pad_to).squeeze()
        feat = pad_feats(torch.Tensor(self.X[idx]).to(
            self.device), max_num_node=self.pad_to)
        target = torch.Tensor([self.y[idx]]).long().to(self.device)
        # add mask for binary
        mask = torch.zeros(adj.shape[0])
        mask[torch.arange(true_nodes)] = 1
        mask = mask.unsqueeze(-1)
        if self.w is not None:
            w = torch.Tensor(self.w[idx]).float().to(self.device)
            return (adj, feat, mask), None, target, w
        else:
            return (adj, feat, mask), None, target
