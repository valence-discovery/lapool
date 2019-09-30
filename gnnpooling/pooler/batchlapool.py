import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from scipy import sparse, cluster, spatial

from gnnpooling.models.layers import FCLayer
from gnnpooling.models.gcn import GINLayer
from gnnpooling.pooler.diffpool import DiffPool

from gnnpooling.utils.tensor_utils import *
from gnnpooling.utils.sparsegen import Sparsegen, Sparsemax
from gnnpooling.utils.graph_utils import find_largest_eigval, compute_deg_matrix, inverse_diag_mat

EPS = 1e-8
LARGE_VAL = 1e4
GraphLayer = partial(GINLayer, eps=0, bias=False, b_norm=False)


def get_path_length(adj_mat, k, strict=True):
    prev_mat = adj_mat
    matlist = adj_mat.unsqueeze(0)
    for i in range(2, k + 1):
        prev_mat = torch.bmm(prev_mat, adj_mat)
        if strict:
            no_path = (matlist.sum(dim=0) != 0).byte()
            new_mat = prev_mat.masked_fill(no_path, 0)
        else:
            new_mat = prev_mat
        # there os no point in receiving msg from the same node.
        # new_mat.clamp_max_(1)
        matlist = torch.cat((matlist, new_mat.unsqueeze(0)), dim=0).clamp_max_(1)
    return matlist


class LaPool(DiffPool):
    r"""
    Attempt to batch laplacian pooling by taking using a selection mask
    """

    def __init__(self, input_dim, cluster_dim, hidden_dim=None, attn='cos', hop=-1, reg_mode=1,
                 concat=False, strict_leader=True, GLayer=GraphLayer, lap_hop=0, sigma=0.8, **kwargs):
        net = GLayer(input_dim, input_dim, activation='relu')
        super(LaPool, self).__init__(input_dim, -1, net)
        self.cur_S = None
        self.leader_idx = []
        self.cluster_dim = cluster_dim
        self.attn_softmax = Sparsegen(dim=-1, sigma=sigma)
        self.attn_net = cosine_attn  # using cosine attention
        if attn == 'dot':
            self.attn_net = dot_attn

        self.concat = concat
        self.feat_update = FCLayer(in_size=input_dim * int(1 + self.concat), out_size=self.cluster_dim,
                                   activation='relu', **kwargs)
        self.k = hidden_dim
        self.reg_mode = reg_mode
        self.hop = hop
        self.lap_hop = lap_hop
        self.strict_leader = strict_leader

    def compute_attention(self, adj, nodes, clusters, mask):
        attn = self.attn_net(nodes, clusters)
        if self.hop >= 0:  # limit to number of hop
            G = get_path_length(adj, self.hop, strict=False).sum(dim=0)  # number of path
            #  torch.sum(torch.stack(                                                                                                                                                                  
            #     [(1/(i))*torch.matrix_power(adj.float(), i).clamp(max=1) for i in range(1, self.hop+1)]), dim=0)
            # force values without a path of at most length hop to not contribute
        else:  # compute full distance
            with np.errstate(divide='ignore'):
                gpath = np.array(
                    [1 / sparse.csgraph.shortest_path(x, directed=False) for x in adj.clone().detach().cpu().numpy()])
            gpath[np.isinf(gpath)] = 0  # normalized distance (higher, better)
            G = to_tensor(gpath, gpu=False, dtype=torch.float).to(adj.device)

        if self.strict_leader:
            G = torch.stack(
                [gg.index_fill(-2, self.leader_idx[ind], 0) + torch.eye(gg.shape[-1]).to(adj.device) for gg, ind in
                 zip(G, range(G.shape[0]))])

        # gives the distance or number of path between each node and the centroid
        last_dim = adj.dim() - 1
        G = batch_index_select(G, last_dim, self.leader_idx) + EPS
        # entry of G should always be zero for non-connected components, so attn will be null for them
        attn = self.attn_softmax(attn * G)
        return attn

    def compute_laplacian(self, adj):
        adj_size = adj.shape[-2]
        deg_mat, adj = compute_deg_matrix(adj)
        if self.reg_mode == 0:
            laplacian = deg_mat - adj  #
        else:
            laplacian = torch.eye(adj_size).to(adj.device) - torch.matmul(inverse_diag_mat(deg_mat), adj)
        # else:
        #     lambda_max = find_largest_eigval(adj)  # might be approximated by max degree
        #     laplacian = (torch.eye(adj_size).to(adj.device) - adj / lambda_max)
        if self.lap_hop > 1:
            laplacian = torch.matrix_power(laplacian, self.lap_hop)
        return laplacian

    def _select_leader(self, adj, x, nodes=None, **kwargs):

        adj_size = adj.shape[-2]
        # Compute the graph Laplacian, then the norm of the Laplacian
        laplacian = self.compute_laplacian(adj).matmul(x)
        laplacian_norm = torch.norm(laplacian, dim=-1)  # b * n

        adj_no_diag = adj - torch.diag_embed(torch.diagonal(adj.permute(1, 2, 0)))
        node_deg, _ = compute_deg_matrix(adj_no_diag, selfloop=False)
        # we want node where the following:
        # \sum(wj xj) / \sum(wj) < xi ==>  D(^-1)AX < X, where A does not have diagonal entry
        nei_laplacian_diff = (
                    laplacian_norm.unsqueeze(-1) - torch.bmm(torch.matmul(inverse_diag_mat(node_deg), adj_no_diag),
                                                             laplacian_norm.unsqueeze(-1))).squeeze(-1)

        # normalize to all strictly positive values
        min_val = torch.min(nei_laplacian_diff, -1, keepdim=True)[0]
        nei_laplacian_normalized = (nei_laplacian_diff - min_val) + torch.abs(min_val)

        if nodes is None:
            nodes = torch.ones_like(nei_laplacian_normalized)

        nei_laplacian_normalized = nei_laplacian_normalized * nodes.squeeze(
            -1).float()  # set unwanted (fake nodes) to 0

        k = self.k
        mask = nei_laplacian_normalized > 0
        if k is None:
            mask = nei_laplacian_diff * nodes.float() > 0
            # find best max k for this batch
            k = torch.max(torch.sum(mask, dim=-1))  # maximum number of valid centroid in the batch
            # note that in this we relax the assumption by computing 
            # \sum\limits_{j \neq i} s_i - a_{ij} s_j \big) > 0, and not
            # \forall\; v_j,  s_i - A_{ij} s_j  > 0  as seen in the paper
        _, leader_idx = torch.topk(nei_laplacian_normalized, k=k, dim=-1, largest=True)  # select k best
        self.leader_idx = leader_idx.to(adj.device)
        return leader_idx, mask

    def compute_clusters(self, *, adj, x, h, nodes, **kwargs):
        leader_idx, mask = self._select_leader(adj, h, nodes)
        clusters = batch_index_select(h, 1, leader_idx)
        attn = self.compute_attention(adj, h, clusters, mask)
        return attn, clusters

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
        # LLP = self.link_loss(adj, mapper) / adj.numel()
        # LE = self.entropy(mapper, dim=-1)
        return 0  # (LLP + LE)

    def compute_adj(self, *, mapper, adj, **kwargs):
        adj = mapper.transpose(-2, -1).matmul(adj).matmul(mapper)
        # Remove diagonal
        adj = (1 - torch.eye(adj.shape[-1]).unsqueeze(0)).to(adj.device) * adj
        return adj

    def compute_feats(self, *, mapper, x, precluster, **kwargs):
        if not self.concat:
            clusters = torch.bmm(mapper.transpose(-2, -1), x)
        else:
            # case where the batchsize is one
            x = x.unsqueeze(0) if x.dim() == 2 else x
            mapper = mapper.unsqueeze(0) if mapper.dim() == 2 else mapper
            precluster = precluster.unsqueeze(0) if precluster.dim() == 2 else precluster
            clusters = torch.cat((torch.bmm(mapper.transpose(-2, -1), x), precluster), dim=-1)
        clusters = self.feat_update(clusters)
        return clusters

    def forward(self, adj, x, mask, return_loss=False, **kwargs):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        _, h = self.net(adj, x)

        S, precluster = self.compute_clusters(adj=adj, x=x, h=h, nodes=mask)

        new_feat = self.compute_feats(mapper=S, x=h, precluster=precluster)

        new_adj = self.compute_adj(mapper=S, adj=adj)

        self.cur_S = S

        if return_loss:
            loss = self._loss(adj, S)
            return new_adj, new_feat, loss

        return new_adj, new_feat
