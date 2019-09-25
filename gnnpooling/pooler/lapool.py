import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnnpooling.models.layers import FCLayer
from gnnpooling.models.gcn import GINLayer
from gnnpooling.pooler.diffpool import DiffPool

from gnnpooling.utils.tensor_utils import *
from gnnpooling.utils.sparsegen import Sparsegen, Sparsemax
from gnnpooling.utils.graph_utils import find_largest_eigval, compute_deg_matrix
from functools import partial
from scipy import sparse, cluster, spatial

EPS = 1e-8
LARGE_VAL = 1e4
GraphLayer = partial(GINLayer, eps=0, bias=False, b_norm=False)

def Spheremax(x, dim=-1):
    return x ** 2 / ((x ** 2).sum(dim=dim, keepdim=True) + EPS)


def Annealedsoftmax(x, T=0.01, dim=-1):
    return F.softmax(x / T, dim=dim)


def _lazy_random_walk(adj):
    deg, adj = compute_deg_matrix(adj, inv=False, selfloop=False)
    deg = deg ** -1
    deg = deg.masked_fill(torch.isinf(deg), 0)
    return torch.matmul(deg, adj)


def get_path_length(adj_mat, k, strict=True):
    prev_mat = adj_mat
    matlist = adj_mat.unsqueeze(0)
    for i in range(2, k + 1):
        prev_mat = torch.mm(prev_mat, adj_mat)
        if strict:
            no_path = (matlist.sum(dim=0) != 0).byte()
            new_mat = prev_mat.masked_fill(no_path, 0)
        else:
            new_mat = prev_mat
        # there os no point in receiving msg from the same node.
        # new_mat.clamp_max_(1)
        matlist = torch.cat((matlist, new_mat.unsqueeze(0)), dim=0).clamp_max_(1)
    return matlist


def get_signal(adj, x):
    deg_mat, adj = compute_deg_matrix(adj.squeeze(0))
    adj_size = adj.size(-1)
    laplacian = torch.eye(adj_size).to(adj.device) - torch.matmul(torch.pinverse(deg_mat), adj)
    return laplacian.matmul(x).norm()


class LaplacianPool(DiffPool):
    r"""
    Differentiable pooling layer trying to merge the topk idea and an affinity matrix idea

    attn_mode = 1 ===> compute attention using cosine similarity
    attn_mode = 2 ===> compute attention using concatenation and a neural network

    Will choose leader (cluster centroids) by sorting them according to largest values. Sigmoid does not have impact here
    hop = None or integer ==> number of k_hop neigbors to consider for each node. If None, consider all molecules

    """

    def __init__(self, input_dim, dropk=None, cluster_dim=None, strict_path=False, attn=1, hop=3, reg_mode=1,
                 concat=False, strict_leader=True, GLayer=GraphLayer, lap_hop=1, sigma=0.5, **kwargs):
        net = GLayer(input_dim, input_dim, activation='relu')
        super(LaplacianPool, self).__init__(input_dim, -1, net)
        self.cur_S = None
        self.alpha = kwargs.pop("alpha", 1)
        self.leader_idx = []
        self.cluster_dim = cluster_dim or self.input_dim
        self.attn_softmax = Sparsegen(dim=-1, sigma=sigma)
        self.attn_mode = attn
        self.strict_path = strict_path
        if attn == 1:
            self.attn_net = cosine_attn  # using cosine attention
        else:
            self.attn_net = nn.Sequential(
                nn.Linear(input_dim * 2, 1),
                nn.LeakyReLU()
            )  # computing base on weight prediction after concatenation
        self.lap_hop = lap_hop
        self.concat = concat
        self.feat_update = FCLayer(in_size=input_dim * int(1 + self.concat), out_size=self.cluster_dim,
                                   activation='relu', **kwargs)
        self.hop = hop
        self.dropk = dropk
        self.reg_mode = reg_mode
        self.strict_leader = strict_leader

    def compute_attention(self, adj, nodes, clusters):
        if self.attn_mode != 1:
            new_tensor = torch.cat([upsample_to(
                nodes, clusters.shape[-2]), upsample_to(clusters, nodes.shape[-2])], dim=-1)
            # torch.cat(nodes.repeat(clusters.shape[0], 1), clusters.repeat(nodes.shape[0], 1)), dim=1)
            attn = self.attn_net(new_tensor)
        else:
            attn = self.attn_net(nodes, clusters)

        attn = attn.view(adj.shape[-1], -1)

        if self.hop:
            if self.hop > 0:
                G = get_path_length(adj, self.hop, strict=self.strict_path).sum(dim=0)
                #  torch.sum(torch.stack(                                                                                                                                                                  
                #     [(1/(i))*torch.matrix_power(adj.float(), i).clamp(max=1) for i in range(1, self.hop+1)]), dim=0)
                # force values without a path of at most length hop to not contribute
            else:
                gpath = sparse.csgraph.shortest_path(adj.clone().detach().cpu().numpy(), directed=False)
                gpath = 1 / gpath
                gpath[np.isinf(gpath)] = 1
                G = to_tensor(gpath, gpu=False, dtype=torch.float).to(adj.device)

            if self.strict_leader:
                G = G.index_fill(-2, self.leader_idx, 0) + torch.eye(adj.shape[-1]).to(adj.device)
            G = torch.index_select(G, -1, self.leader_idx) + EPS

            attn = self.attn_softmax(attn * G)
            return attn

        return self.attn_softmax(attn)

    def compute_laplacian(self, adj):
        adj_size = adj.shape[-2]
        deg_mat, adj = compute_deg_matrix(adj)
        if self.reg_mode == 0:
            laplacian = deg_mat - adj  #
        elif self.reg_mode == 1:
            laplacian = torch.eye(adj_size).to(adj.device) - torch.matmul(torch.pinverse(deg_mat), adj)
        else:
            lambda_max = find_largest_eigval(adj)  # might be approximated by max degree
            laplacian = (torch.eye(adj_size).to(adj.device) - adj / lambda_max)
        if self.lap_hop > 1:
            laplacian = torch.matrix_power(laplacian, self.lap_hop)
        return laplacian

    def _select_leader(self, adj, x, **kwargs):
        # if self.topk_mode == 1:  
        # No batches
        adj_size = adj.shape[-2]
        adj = adj.squeeze(0)
        x = x.squeeze(0)
        # Compute the graph Laplacian, then the norm of the Laplacian
        laplacian = self.compute_laplacian(adj).matmul(x)
        laplacian_norm = torch.norm(laplacian, dim=-1)
        if self.dropk is not None:
            k = self.dropk
            if 0 < k < 1:
                k = max(1, int(np.floor((1 - k) * adj_size)))
            _, leader_idx = torch.topk(laplacian_norm, k=k, dim=-1, largest=True)

        else:
            # Select the leaders as the nodes with a Laplacian greater or equal then all its neighbours
            max_val, _ = torch.max(laplacian_norm.squeeze() * adj, dim=-1)
            # TODO: find a way to find indexes of laplacian_norm - 0.999*max_val
            leader_idx = torch.masked_select(torch.arange(0, adj_size), laplacian_norm > (1 - 1e-4) * max_val)

            # check if all neighbors are also leaders
        true_leader_idx = ~(torch.sum(adj[leader_idx], dim=-1) ==
                           torch.sum(adj[leader_idx, :][:, leader_idx], dim=-1))
        num_leader_filtered = true_leader_idx.sum()
        # If no leader due to filtering, choose the first atom as a leader
        if num_leader_filtered == 0:
            leader_idx = torch.Tensor([0]).long()
            true_leader_idx = torch.Tensor([1]).long()

        self.leader_idx = torch.masked_select(leader_idx.to(adj.device), true_leader_idx.byte())

    def compute_graph_energy(self, ori_input, new_input):
        ori_signal = get_signal(*ori_input)
        new_signal = get_signal(*new_input)
        return ori_signal, new_signal

    def compute_clusters(self, *, adj, x, h, **kwargs):
        self._select_leader(adj, x)
        clusters = torch.index_select(h, 1, self.leader_idx)
        attn = self.compute_attention(adj.squeeze(), h, clusters)
        return attn.unsqueeze(0), clusters

    @staticmethod
    def entropy(S, dim):
        return (-S * torch.log(S.clamp(min=EPS))).sum(dim=dim).mean()


    @staticmethod
    def link_loss(adj, mapper):
        return torch.norm(adj - torch.matmul(mapper, mapper.transpose(1, 2)), p=2)


    def _loss(self, adj, mapper):
        r"""
        Compute the auxiliary link prediction loss for the given batch

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Adjacency matrix of size (B, N, N)
            mapper: `torch.FloatTensor`
                Node assignment matrix of size (B, N, M)
        """
        LLP = self.link_loss(adj, mapper) / adj.numel()
        LE = self.entropy(mapper, dim=-1)
        return (LLP + LE)


    def compute_adj(self, *, mapper, adj, **kwargs):
        adj = mapper.transpose(-2, -1).matmul(adj).matmul(mapper)
        # Remove diagonal
        adj =  (1 - torch.eye(adj.shape[-1]).unsqueeze(0)).to(adj.device) * adj
        return adj


    def compute_feats(self, *, mapper, x, precluster, **kwargs):
        if not self.concat:
            clusters = torch.mm(mapper.t(), x) + getattr(self, "alpha", 1) * precluster
        else:
            mapper = mapper.index_fill_(-2, self.leader_idx, 0)
            clusters = torch.cat((torch.mm(mapper.t(), x), precluster), dim=-1)
        clusters = self.feat_update(clusters)
        return clusters.unsqueeze(0)

    def forward(self, adj, x, return_loss=False, **kwargs):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        _, h = self.net(adj, x)
        S, precluster = self.compute_clusters(adj=adj, x=x, h=h)
        new_feat = self.compute_feats(mapper=S.squeeze(
            0), x=h.squeeze(0), precluster=precluster.squeeze(0))
        new_adj = self.compute_adj(mapper=S, adj=adj)
        self.cur_S = S
        if return_loss:
            loss = self._loss(adj, S)
            return new_adj, new_feat, loss
        return new_adj, new_feat
