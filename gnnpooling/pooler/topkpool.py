import torch
from torch import nn
from gnnpooling.pooler.base import GraphPool
from gnnpooling.utils.tensor_utils import batch_index_select, upsample_to


class TopKPool(GraphPool):
    r"""
    This layer implements the  :math:`\mathrm{top}_k` pooling operator from the `"Graph U-Net"
    <https://openreview.net/forum?id=HJePRoAct7>`_ and `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_ papers

    .. math::
        \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{y^{\prime}} &=  \mathrm{tanh}(\mathbf{y}(i))


        \mathbf{X}^{l+1} &= (\mathbf{X^{l}}(i) \odot \mathbf{y^{\prime}})

        \mathbf{A}^{{l+1}} &= \mathbf{A^{l+1}}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Arguments
    ---------
        input_dim : int
            Expected input dimension of the layer. This corresponds to the size of embedded features.
        hidden_dim: int
            Hidden and output dimension of the layer. This is the maximum number of clusters to which nodes will be assigned.

    Attributes
    ----------
        input_dim: int
            Input dim size
        hidden_dim: int
            Maximum number of cluster
        cur_S: Union[`torch.Tensor`, None]
    """

    def __init__(self, input_dim, hidden_dim):

        super(TopKPool, self).__init__(input_dim, hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.Tensor(1, self.input_dim))
        self.net = torch.topk
        self.cur_S = None
        self.cur_idx = None
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize or reset the parameters of the network"""
        torch.nn.init.xavier_uniform_(self.weight)

    def _filter_adj(self, adj, idx):
        """Filter adjacency list to keep only input index"""
        if idx.dim() == 2:
            new_adj = batch_index_select(
                batch_index_select(adj, 2, idx), 1, idx)
        else:
            new_adj = torch.index_select(adj, -2, idx).index_select(-1, idx)
        return new_adj

    def compute_clusters(self, adj, x, **kwargs):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        see `GraphPool.compute_clusters`
        """
        score = (x * self.weight).sum(dim=-1)
        score = score / self.weight.norm(p=2, dim=-1)
        _, idx = self.net(score, k=self.hidden_dim, dim=1, largest=True)
        # slightly change how this is changed
        return idx, score

    def forward(self, adj, x, **kwargs):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Unormalized adjacency matrix of size (B, N, N)
            x: torch.FloatTensor 
                Node feature after embedding of size (B, N, D). See paper for more infos

        Returns
        -------
            new_adj: `torch.FloatTensor`
                New adjacency matrix, describing the reduced graph
            new_feat: `torch.FloatTensor`
                Node features resulting from the pooling operation.
            idx: `torch.LongTensor`, optional
                Selected layer for the tensor.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """

        x = x.unsqueeze(0) if x.dim() == 2 else x
        # we always want batch computation
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        idx, score = self.compute_clusters(adj, x)
        self.cur_idx = idx
        new_feat = torch.mul(x, torch.tanh(score.unsqueeze(-1)))
        new_feat = new_feat[idx] if idx.dim() == 1 else batch_index_select(new_feat, 1, idx)
        self.cur_S = torch.zeros(adj.shape[-1], len(idx.flatten()))
        self.cur_S[idx, torch.arange(self.hidden_dim)] = 1
        new_adj = self._filter_adj(adj, idx)
        if kwargs.pop("return_loss", False): 
            return new_adj, new_feat, 0
        return new_adj, new_feat
