import torch
import torch.nn as nn
import torch.nn.functional as F
from gnnpooling.models.gcn import GCNLayer, GINLayer
from gnnpooling.models.layers import FCLayer
from .base import GraphPool
import networkx as nx

EPS = 1e-7


class DiffPool(GraphPool):
    r"""
    This layer implements the differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling" <https://arxiv.org/abs/1806.08804>`_ paper.
    It computes a learned assignment :math:`\mathbf{S_l} \in \mathbb{R}^{B \times N_{l-1} \times N_{l} }`.

    .. math::
        \mathbf{S_{l+1}} = \mathrm{softmax}(\left( GCN_{l,pool}(A_l, X_l)\right)

    Then the coarsed adjacency matrix :math:`A_{l+1}` and the pooled node features :math:`X_{l+1}` can be computed as follow: 

    .. math ::

        \mathbf{A}_{l+1} &= \mathbf{S}^{\top} \cdot \mathbf{A_l} \cdot \mathbf{S}
        \mathbf{X}_{l+1} &= \mathbf{S}^{\top} \cdot \mathbf{Z_l}

    where :math:`Z_l` is an embedding of the input features by graphconv.
    The layer can also return the auxiliary link prediction objective :math:`|| \mathbf{A}_l - \mathbf{S}_l {\mathbf{S}_l}^{\top} ||_F`

    Arguments
    ---------
        input_dim : int
            Expected input dimension of the layer. This corresponds to the size of embedded features
        hidden_dim: int
            Hidden and output dimension of the layer. This is the maximum number of clusters to which nodes will be assigned.
        net: `nn.Module`, optional
            Network for computing :math:`S_l`. If None, the default GaphSage computation 
        kwargs: 
            named parameters for the node assignment network

    Attributes
    ----------
        input_dim: int
            Input dim size
        hidden_dim: int
            Maximum number of cluster
        net: `nn.Module`
            Network for assigning nodes to clusters
        cur_S: Union[`torch.Tensor`, None]
            Current value of the assignment matrix after seeing the last batch.
    """

    def __init__(self, input_dim, hidden_dim, net=None, **kwargs):

        super(DiffPool, self).__init__(input_dim, hidden_dim, net)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        def_param = dict(bias=False, b_norm=False, normalize=True)
        def_param.update(kwargs)
        if self.net is None:
            self.net = GINLayer(input_dim, hidden_dim, **def_param)

    def compute_clusters(self, *, adj, x, **kwargs):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        see `GraphPool.compute_clusters`
        """
        G, h = self.net(adj, x)
        S = F.softmax(h, dim=-1)
        return S

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
        r"""
        Applies the Pooling layer on input tensor x and adjacency matrix, to get the pooled adjacency matrix

        Arguments
        ---------
            mapper: `torch.FloatTensor`
                The matrix transformation that maps the adjacency matrix to a new one
            x: `torch.FloatTensor`
                Node feature input tensor before embedding of size (B, N, F)
            kwargs:
                Named parameters to be used by the 
        Returns
        -------
            adj: `torch.FloatTensor`
                The adjacency matrix of the reduced graph
        """
        adj = mapper.transpose(-2, -1).matmul(adj).matmul(mapper)
        # Remove diagonal
        adj = (1 - torch.eye(adj.shape[-1]).unsqueeze(0)) * adj

        return adj


    def compute_feats(self, *, mapper, x, **kwargs):
        r"""
        Applies the Pooling layer on input tensor x and adjacency matrix , to get the pooled features

        Arguments
        ---------
            mapper: `torch.FloatTensor`
                The matrix transformation that maps the adjacency matrix to a new one
            x: `torch.FloatTensor`
                Node feature input tensor before embedding of size (B, N, F)
            kwargs:
                Named parameters to be used by the 
        Returns
        -------
            h: `torch.FloatTensor`
                Node feature for the reduced graph
        """
        return torch.matmul(mapper.transpose(-2, -1), x)

    def forward(self, adj, z, x=None, return_loss=False, **kwargs):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Unormalized adjacency matrix of size (B, N, N)
            z: torch.FloatTensor 
                Node feature after embedding of size (B, N, D). See paper for more infos
            x: torch.FloatTensor, optional
                Input node feature of the convolution layer of size (B, N, F). 
                If None, the embedded features is used.
            return_loss: bool, optional
                Whether to return the auxiliary link prediction loss

        Returns
        -------
            new_adj: `torch.FloatTensor`
                New adjacency matrix, describing the reduced graph
            new_feat: `torch.FloatTensor`
                Node features resulting from the pooling operation.
            loss: `torch.FloatTensor`, optional
                optional edge prediction loss for the layer

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """

        # we always want batch computation
        z = z.unsqueeze(0) if z.dim() == 2 else z
        if x is None:
            x = z
        else:
            x= x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj 
        S = self.compute_clusters(adj=adj, x=x)
        new_feat = self.compute_feats(mapper=S, x=z)
        new_adj = self.compute_adj(mapper=S, adj=adj)
        self.cur_S = S
        if return_loss:
            loss = self._loss(adj, S)
            return new_adj, new_feat, loss
        return new_adj, new_feat
