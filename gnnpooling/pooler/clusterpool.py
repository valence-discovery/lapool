import torch
import numpy as np
from scipy import sparse, cluster, spatial
from functools import partial
from gnnpooling.models.layers import FCLayer
from gnnpooling.models.gcn import GINLayer, GCNLayer
from gnnpooling.pooler.base import GraphPool, _cluster_to_matrix, _cluster_from_list, _get_clustering_pool
try:
    from torch_cluster import graclus_cluster
except ImportError:
    graclus_cluster = None

LARGE_VAL = 1e8
eps = 1e-8


def graclus(adj, x, nclusters):
    r"""
    The greedy clustering algorithm from the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper of picking an unmarked
    vertex and matching it with one of its unmarked neighbors (that maximizes
    its edge weight).

    Arguments
    ---------
        adj: `torch.FloatTensor` of size (N, N)
            Adjacency matrix, will be converted to sparse if it is not sparse
        x : `torch.FloatTensor` of size (N, D)
            Node feature embedding in D dimension
        nclusters: int
            Number of desired clusters. This is useless for graclus as it choose this number itself

    :rtype: :class:`LongTensor`
    """
    if not adj.is_sparse:
        adj = adj.to_sparse()
    row, col = adj.indices()
    weight = adj.values()
    if graclus_cluster is None:
        raise NotImplementedError(
            "Graclus clsutering is not available, please use another clustering method")
    clusters = graclus_cluster(row, col, weight)
    return _cluster_to_matrix(clusters, nclusters=nclusters)


def dggc(adj, x):
    r"""
    The downsampling algorithm described in `"Deep Geometrical Graph Classification" <https://openreview.net/pdf?id=Hkes0iR9KX>`_ 

    Arguments
    ---------
        adj: `torch.FloatTensor` of size (N, N)
            Adjacency matrix, will be converted to sparse if it is not sparse
        x : `torch.FloatTensor` of size (N, D)
            Node feature embedding in D dimension

    :rtype: :class:`LongTensor`
    """
    n = adj.shape[-1]
    triu_val = torch.clamp(torch.nn.functional.pdist(x), 0.0, np.inf)**2
    D = torch.zeros_like(adj)
    D[np.triu_indices(n, 1)] = triu_val
    D = D + D.t()  # adding symmetry back
    D[torch.eye(n).byte()] = np.inf  # we do not want the diagonal
    clusters = []
    selected = torch.ones(n).byte()
    for k in range(0, n//2):
        i, j = np.unravel_index(torch.argmin(D), D.shape)
        clusters.append([i, j])
        D[(i, j), :] = np.inf
        D[:, (j, i)] = np.inf
        selected[i] = selected[j] = 0
    if n % 2 == 1:  # missing node to be added
        last_i = torch.nonzero(selected).unsqueeze(0)
        clusters.append([last_i])
    return _cluster_from_list(clusters, device=adj.device)


def hierachical(adj, x, nclusters, method="single", metric="sqeuclidean", gweight=0.5, edge_penalty=False):
    r"""
    Hierachical clustering algorithm using the node features. We use a distance trick to increase the distance between 
    two node according to the length of the shortest path between them from the adjacency matrix.

    Essentially, the computed distance is :math:`\alpha D_{path} +(1-\alpha) D_{feats}` where :math:`\alpha` is the weight of the graph layout contribution
    towards the computed distance.

    .. warning::
        This distance function can be slow ...

    Arguments
    ---------
        adj: `torch.FloatTensor` of size (N, N)
            Adjacency matrix, will be converted to sparse if it is not sparse
        x : `torch.FloatTensor` of size (N, D)
            Node feature embedding in D dimension
        nclusters: int
            Number of cluster desired.
        method : str, optional
            Linkage method (one of {'single', 'complete', 'average', 'weighted', 'ward', 'median', 'centroid'}
            (Default value = 'single')      
        metric : str, optional
            Distance metric to use between clusters
            (Default value = 'sqeuclidean')   
        gweight: int, optional
            Weight of the graph structure contribution toward clustering.
            (Default value = 0.5)
        edge_penalty: bool, optional
            If set to true, distance between nodes that do not have an edge in the graph will be set to a large value
            (Default value = False)
    :rtype: :class:`LongTensor`
    """
    adj_mat = adj.data.numpy()
    gpath = sparse.csgraph.shortest_path(adj_mat, directed=False)
    gpath[np.isinf(gpath)] = LARGE_VAL
    # normalization of the shortest path
    #gpath = gpath  / np.linalg.norm(gpath)
    # actually make sense to normalize by the max length
    #gpath = gpath / gpath.max()
    i, j = np.triu_indices_from(gpath, k=1)
    gdist = gpath[i, j]
    gdist = (gdist - np.min(gdist[gdist!=LARGE_VAL])) / (np.max(gdist[gdist!=LARGE_VAL])- np.min(gdist[gdist!=LARGE_VAL] + eps))

    edges = (adj_mat[i, j] == 0)
    xdist = spatial.distance.pdist(x.data.numpy(), metric=metric)
    xdist = (xdist - np.min(xdist)) / (np.max(xdist)- np.min(xdist) + eps)
    dist = gweight*gdist + (1-gweight)*xdist
    if edge_penalty:
        dist[edges] = LARGE_VAL
    z = cluster.hierarchy.linkage(dist, method=method, metric=metric)
    clusters = cluster.hierarchy.fcluster(z, nclusters, criterion='maxclust')
    return _cluster_to_matrix(clusters, device=adj.device, nclusters=nclusters)


def kmean(adj, x, nclusters):
    r"""
    The downsampling algorithm described in `"Deep Geometrical Graph Classification" <https://openreview.net/pdf?id=Hkes0iR9KX>`_ 
    gave me other ideas that are implemented below

    Arguments
    ---------
        adj: `torch.FloatTensor` of size (N, N)
            Adjacency matrix, will be converted to sparse if it is not sparse
        x : `torch.FloatTensor` of size (N, D)
            Node feature embedding in D dimension
        nclusters : `torch.FloatTensor` of size (N, D)
            Number of desired clusters

    :rtype: :class:`LongTensor`
    """

    _, clusters = cluster.vq.kmeans2(x.data.numpy(), nclusters, minit='points')
    return _cluster_to_matrix(clusters, nclusters=nclusters)


class ClusterPool(GraphPool):
    r"""
    This pooling layers implements deterministic pooling based of any clustering algorithm

    Arguments
    ---------
        input_dim : int
            Expected input dimension of the layer. This corresponds to the size of embedded features
        hidden_dim: int
            Hidden and output dimension of the layer. This is the maximum number of clusters to which nodes will be assigned.
            Alternatively, you can provide the percentage of atom in the molecule that should be retained as new cluster
        algo: str, optional
            Name of the clustering algorithm to be used. 
            (Default value = "graclus")
        pooling: str, optional
            Cluster feature to be used. By default, the weight sum given by the node mapper matrix transformation will be used.
            Other accepted values are `{'w-sum', sum', 'max', 'avg'}`. The weighted sum is used by default.
            (Default value = 'w-sum')
        clip: bool, optional
            Whether to clip the adjacency matrix, to only have 0-1 values
            (Default value = True)
        kwargs: 
            named parameters for the function computing the node assignment

    .. warning::
            The `graclust` method can potentially create node without any edges (several connected component). 
            There is no way to prevent that, as it chooses its optimal number of cluster. Therefore, clusters are splits or merged until the correct size is obtained.

    Attributes
    ----------
        input_dim: int
            Input dim size
        hidden_dim: int
            Maximum number of cluster or percentage of the graph to retain
        net: Union[callable, `torch.nn.Module`]
            Network for assigning nodes to clusters
        cur_S: Union[`torch.Tensor`, None]
            Current value of the assignment matrix after seeing the last batch.
    """
    CLUSTERING_ALGO = {"graclus": graclus, 'kmean':kmean, 'hierachical': hierachical}

    def __init__(self, input_dim, hidden_dim, algo="kmean", pooling="w-sum", clip=True, **kwargs):
        if 0 <= hidden_dim <= 1:
            hdim = -1
        else:
            hdim = hidden_dim        
        super(ClusterPool, self).__init__(input_dim, hdim)
        self.algo = algo
        self.pooling = _get_clustering_pool(pooling)
        self.clip = clip
        self.net = partial(self.CLUSTERING_ALGO[self.algo], **kwargs)
        self.keep_k = hidden_dim if hdim==-1 else 0
        self.gin = GINLayer(in_size=self.input_dim, kernel_size=self.input_dim)

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
        
        _, x = self.gin(adj, x)
        x = x.squeeze(0)
        mol_size = self.hidden_dim
        if self.keep_k: 
            mol_size = int(np.ceil(self.keep_k * adj.shape[-1]))
        return self.net(adj, x, nclusters=mol_size)


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
        if self.pooling is not None:
            return self.pooling(mapper, x)
        return torch.matmul(mapper.transpose(-2, -1), x)

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
        x = x.unsqueeze(0) if x.dim() == 2 else x  # we always want batch computation
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # batch, oh yeaahhh
        new_adj_list = []
        new_feat_list = []
        for k in range(x.shape[0]):
            S = self.compute_clusters(adj=adj[k], x=x[k], **kwargs)
            new_feat = self.compute_feats(mapper=S, adj=adj[k], x=x[k], **kwargs)
            new_adj = (S.transpose(-2, -1)).matmul(adj[k]).matmul(S)
            self.cur_S = S
            if self.clip:
                new_adj = torch.clamp(new_adj, max=1)
            new_adj_list.append(new_adj)
            new_feat_list.append(new_feat)

        return torch.stack(new_adj_list), torch.stack(new_feat_list)


class DGGC(ClusterPool):
    r"""
    This pooling layers implements the clustering algorithm described in `"Deep Geometrical Graph Classification" <https://openreview.net/pdf?id=Hkes0iR9KX>`_ .
    The output dim of this layer is fixed and corresponds to half of the input

    Arguments
    ---------
        input_dim : int
            Expected input dimension of the layer. This corresponds to the current size of the graph.
        clip: bool, optional
            Whether to clip the adjacency matrix, to only have 0-1 values
            (Default value = True)
        pooling: str, optional
            Cluster feature to be used. By default, the weight sum given by the node mapper matrix transformation will be used.
            Other accepted values are `{'w-sum', sum', 'max', 'avg'}`. The weighted sum is used by default.
            (Default value = 'w-sum')

    Attributes
    ----------
        input_dim: int
            Input dim size
        hidden_dim: int
            Number of cluster returned by this layer
        net: callable
            Function for assigning nodes to clusters
    """
    def __init__(self, input_dim, pooling="w-sum", clip=True):
        super(ClusterPool, self).__init__(input_dim, 0)
        self.hidden_dim = (input_dim // 2) + (input_dim % 2) # the layer halves
        self.pooling = _get_clustering_pool(pooling)
        self.clip = clip        
        self.net = dggc

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
        return self.net(adj, x)