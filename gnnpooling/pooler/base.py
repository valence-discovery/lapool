import torch
from torch import nn


def sum_pool_cluster(clustermap, x, dim=0):
    """Compute sum pooling for nodes in a given cluster"""
    # Using the hard-clip approach instead
    # Please note that this is not differentiable as there is no autograd for indexing.

    # nsize = clustermap.shape[-1]
    # x_new = torch.cat([torch.sum(x[clustermap[:, k].byte()], keepdim=True, dim=dim) for k in range(nsize)])
    # return x_new
    row_max = _matrix_to_cluster(clustermap)
    uniq = torch.unique(row_max, sorted=True)
    x_new = torch.cat([torch.sum(x.index_select(dim, (row_max == k).nonzero(
    ).squeeze().long()), keepdim=True, dim=dim) for k in uniq])
    return x_new


def avg_pool_cluster(clustermap, x, dim=0):
    """Compute avg pooling for nodes in a given cluster"""
    row_max = _matrix_to_cluster(clustermap)
    uniq = torch.unique(row_max, sorted=True)
    x_new = torch.cat([torch.mean(x.index_select(dim, (row_max == k).nonzero(
    ).squeeze().long()), keepdim=True, dim=dim) for k in uniq])
    return x_new


def max_pool_cluster(clustermap, x, dim=0):
    """Compute max pooling for nodes in a given cluster"""
    row_max = _matrix_to_cluster(clustermap)
    uniq = torch.unique(row_max, sorted=True)
    x_new = torch.cat([torch.max(x.index_select(dim, (row_max == k).nonzero(
    ).squeeze().long()), keepdim=True, dim=dim) for k in uniq])
    return x_new


def _get_clustering_pool(pool):
    """Get the pooling function to use for the cluster pooling"""
    if not pool:
        return None
    elif pool.lower() == "sum":
        return sum_pool_cluster
    elif pool.lower() == "max":
        return max_pool_cluster
    elif pool.lower() in ["avg", "mean"]:
        return avg_pool_cluster
    return None


def _matrix_to_cluster(mat):
    """Convert a NxM assignment matrix to a cluster list of length N"""
    assert mat.dim() == 2, "Expected a 2D matrix, got {}".format(mat.shape)
    return mat.argmax(dim=1)


def __merge_until(clusters, n):
    """Select and merge two clusters until we have n"""
    cur_number = len(torch.unique(clusters))
    while cur_number > n:
        lowest_two, _ = torch.topk(torch.unique(clusters), k=2, largest=False)
        max_id = torch.max(lowest_two)
        min_id = torch.min(lowest_two)
        clusters[clusters == min_id] = max_id
        cur_number -= 1
    return clusters


def __split_until(clusters, n):
    """Select and split one clusters, until we have n"""
    max_number = torch.max(clusters)
    cur_number = len(torch.unique(clusters))
    while cur_number < n:
        largest = torch.argmax(torch.bincount(clusters))
        inds = torch.nonzero(clusters == int(largest)).squeeze()
        half_index = torch.tensor([inds[i]
                                   for i in range(len(inds)//2)]).squeeze()
        # print(half_index)
        # print(inds)
        clusters[half_index] = max_number + 1
        max_number += 1
        cur_number += 1
    return clusters


def _cluster_preprocess(src):
    r"""Preprocess a mapping from node index to cluster index"""
    uniq, clusters = torch.unique(src, sorted=True, return_inverse=True)
    return clusters, len(uniq)


def _cluster_to_matrix(clusters, device=None, nclusters=None):
    r"""Preprocess a list of cluster to transform them into an assignment matrix"""
    if isinstance(clusters, torch.Tensor) and device is None:
        device = clusters.device
    if not isinstance(clusters, torch.Tensor):
        clusters = torch.tensor(clusters).int()
    n = len(torch.unique(clusters))
    nclusters = nclusters or n
    if nclusters > n:
        clusters = __split_until(clusters, nclusters)
    elif nclusters < n:
        clusters = __merge_until(clusters, nclusters)
    clusters, n = _cluster_preprocess(clusters)
    mapper = torch.zeros(nclusters, len(clusters), device=device).scatter_(
        0, clusters.unsqueeze(0), 1).t()
    return mapper


def _cluster_from_list(cluster_list, device=None, nclusters=None):
    r"""Transform a list of list of cluster into an assignment matrix"""
    ncluster = nclusters or len(cluster_list)
    nnodes = sum([len(x) for x in cluster_list])
    mapper = torch.zeros(nnodes, ncluster, device=device)
    for k, cluster in enumerate(cluster_list):
        mapper[cluster, k] = 1
    return mapper


class GraphPool(nn.Module):
    r"""
    This layer implements an abstract class for graph pooling, and should be the base class of any graph pooling layer.

    Arguments
    ---------
        input_dim : int
            Expected input dimension of the layer. This corresponds to the size of embedded features :math:`Z_l` 
        hidden_dim: int
            Hidden and output dimension of the layer. This is the maximum number of clusters to which nodes will be assigned.

    Attributes
    ----------
        input_dim: int
            Input dim size
        hidden_dim: int
            Maximum number of cluster
        net: Union[`torch.nn.Module`, callable]
            Network or function that computes the transformation to clusters
       cur_S: Union[`torch.Tensor`, None]
            Current value of the assignment matrix after seeing the last batch.
    """

    def __init__(self, input_dim, hidden_dim, net=None, **kwargs):
        super(GraphPool, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = net
        self.cur_S = None

    @property
    def output_dim(self):
        r"""
        Dimension of the output feature space in which the sequences are projected

        Returns
        -------
            output_dim: int
                Dimension of the feature space

        """
        return self.hidden_dim

    def compute_clusters(self, *, adj, x, **kwargs):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Adjacency matrix of size (B, N, N)
            x: torch.FloatTensor 
                Node feature input tensor containing node embedding of size (B, N, F)
            kwargs:
                Named parameters to be passed to the function computing the clusters
        """
        raise NotImplementedError(
            "This function should be overriden by your pooling layer")

    def _loss(self, *args):
        r"""
        Compute a specific loss for the layer.

        Arguments
        ----------
            *args: 
                Arguments for computing the loss
        """
        raise NotImplementedError(
            "This function should be overriden by your pooling layer")

    def compute_feats(self, *, mapper, adj, x, **kwargs):
        r"""
        Applies the Pooling layer on input tensor x and adjacency matrix 

        Arguments
        ---------
            mapper: `torch.FloatTensor`
                The matrix transformation that maps the adjacency matrix to a new one
            adj: `torch.FloatTensor`
                Unormalized adjacency matrix of size (B, N, N)
            x: `torch.FloatTensor`
                Node feature input tensor before embedding of size (B, N, F)
            kwargs:
                Named parameters to be used by the 
        Returns
        -------
            h: `torch.FloatTensor`
                Node feature for the reduced graph
        """
        raise NotImplementedError(
            "This function should be overriden by your pooling layer")

    def forward(self, adj, x, **kwargs):
        r"""
        Applies the Pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Unormalized adjacency matrix of size (B, N, N)
            x: torch.FloatTensor 
                Node feature input tensor before embedding of size (B, N, F)
            kwargs:
                Named parameters to be used by the forward computation graph
        Returns
        -------
            new_adj: `torch.FloatTensor`
                New adjacency matrix, describing the reduced graph
            new_feat: `torch.FloatTensor`
                Node features resulting from the pooling operation.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x 
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        S = self.compute_clusters(adj=adj, x=x, **kwargs)
        self.cur_S = S
        new_feat = self.compute_feats(mapper=S, adj=adj, x=x, **kwargs)
        new_adj = (S.transpose(-2, -1)).matmul(adj).matmul(S)
        return new_adj, new_feat
