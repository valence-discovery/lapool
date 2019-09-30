import torch
import dgl
from functools import partial
from torch import nn
from gnnpooling.models.layers import get_pooling, get_activation, FCLayer
from gnnpooling.utils.graph_utils import pack_graph, normalize_adj, compute_deg_matrix, adj_mat_from_edges, dgl_from_edge_matrix


class GCNLayer(nn.Module):
    r"""
    Standard Graph Convolution Layer, i

    .. warning::
        If you fix the input graph size, and set `pack_batch=True` please normalize before ! Otherwise, this will be really slow ...

    Arguments
    ----------
        input_size: int
            Input size of the node feature. Note that this will be checked against the input features
            during each forward pass.
        kernel_size: int
            The size of the node feature vectors in the output graph.
        G_size (int, optional): Size of the square adjacency matrix. If this is not provided, Graph packing will be assumed.
            (Default value = None)
        bias: bool, optional
            Whether to use bias in the linear transformation
            (Default value = False)
        normalize: bool, optional
            whether the input adjacency should be normalized. Use this if input adjacency was not previously normalized
            (Default value = True)
        **kwargs: named parameters
            Additionnal named parameters to be passed to `GraphConvLayer`
            (e.g: activation, dropout, b_norm, pooling, pack_batch)

    Attributes
    ----------
        in_size: int
            size of the input feature space
        out_size: int
            size of the output feature space
        G_size: int
            number of elements (atom) per graphs
        net: torch.nn.Linear
            dense layer to project input feature
        pack_batch: bool
            Whether to pack the batch of graphs
        normalize: bool
            Whether to normalize input

    """

    def __init__(self, in_size, kernel_size, G_size=None, bias=False, normalize=False, activation='relu', dropout=0., b_norm=False, pooling='sum', pack_batch=False):
        # Accepted methods for pooling are avg, max and sum
        super().__init__()
        self.in_size = in_size
        self.out_size = kernel_size
        self.G_size = G_size
        self._pooling = get_pooling(pooling)
        self.pack_batch = pack_batch
        self.net = FCLayer(self.in_size, self.out_size, activation=activation,
                           bias=bias, dropout=dropout, b_norm=b_norm)
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        pass
        
    def gather(self, h, nodes_per_graph=None):
        if self.pack_batch:
            if not nodes_per_graph:
                raise ValueError("Expect node_per_mol for packed graph")
            return torch.squeeze(torch.stack([self._pooling(mol_feat)
                                              for mol_feat in torch.split(h, nodes_per_graph, dim=1)], dim=0), dim=1)
        return torch.squeeze(self._pooling(h), dim=1)

    def _forward(self, h):
        # reshape to fit linear layer requirements
        h = h.view(-1, self.in_size)
        h = self.net(h)
        return h

    def forward(self, G, x, mask=None):
        G_size = self.G_size
        if not self.pack_batch and isinstance(G, (list, tuple)):
            G = torch.stack(G)
            x = torch.stack(x)  # .requires_grad_()

        if not isinstance(G, torch.Tensor) and self.pack_batch:
            G, h = pack_graph(G, x, False)
            G_size = h.shape[0]

        else:  # expect a batch here
            # ensure that batch dim is there
            xshape = x.shape[2] if x.dim() > 2 else x.shape[1]
            G = G.view(-1, G.shape[-2], G.shape[-1])
            h = x.view(-1, G.shape[1], xshape)
            G_size = h.shape[-2]

        if self.normalize:
            norm_G = normalize_adj(G)
            h = torch.matmul(norm_G, h)  # batch_size, G_size, in_size
        else:
            h = torch.matmul(G, h)
        h = self._forward(h)
        # set the batch dimension again for normal use
        h = h.view(-1, G_size, self.out_size)
        if mask is not None:
            h = h * mask.view(-1, G_size, 1).to(h.dtype)
        return G, h

    @property
    def output_dim(self):
        return self.out_size


class GINLayer(nn.Module):
    r"""
    This layer implements the graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{t+1}_i = NN_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x^{t}}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x^{t+1}}_j \right),

    where :math:`NN_{\mathbf{\Theta}}` denotes a neural network.

    Arguments
    ---------
        in_size: int
            Input dimension of the node feature
        kernel_size : int
            Output dim of the node feature embedding
        eps : float, optional
            (Initial) :math:`\epsilon` value. If set to None, eps will be trained 
            (Default value = None)
        net : `nn.Module`, optional
            Neural network :math:`NN_{\mathbf{\Theta}}`. FCLayer by default
            (Default value = None)
        pooling: str or callable, optional
            Pooling function to use. 
            (Default value = 'sum')
        pack_batch: bool, optional
            Whether to pack the batch of graph into a larger one.
            Use this if the batch of graphs have various size.
            (Default value = False)
        kwargs:
            Optional named parameters to send to the neural network
    """

    def __init__(self, in_size, kernel_size, G_size=None, eps=None, net=None, pooling="sum", pack_batch=False, **kwargs):

        super(GINLayer, self).__init__()
        self.in_size = in_size
        self.out_size = kernel_size
        self._pooling = get_pooling(pooling)
        self.pack_batch = pack_batch
        self.G_size = G_size
        kwargs.pop("normalize", None)
        self.net = (net or FCLayer)(in_size, kernel_size, **kwargs)
        self.chosen_eps = eps
        if eps is None:
            self.eps = torch.nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        r"""
        Initialize weights of the models, as defined by the input initialization scheme.

        Arguments
        ----------
            init_fn (callable, optional): Function to initialize the linear weights. If it is not provided
                an attempt to use the object `self.init_fn` would be first made, before doing nothing.
                (Default value = None)

        See Also
        --------
            `nn.init` for more information
        """
        chosen_eps = self.chosen_eps or 0
        if not isinstance(self.eps, nn.Parameter):
            self.eps.data.fill_(chosen_eps)

    def gather(self, h, nodes_per_graph=None):
        r"""
        Graph level representation of the features. 
        This function apply the pooling layer to gather and aggregate information on all nodes

        Arguments
        ----------
            h: torch.FloatTensor of size B x N x M of P x M
                Learned features for all atoms for each graph.
                If the graph is not packed, the first dimension should correspond to the batch size (B).
            node_per_graph: list, optional
                If the graph is packed, this argument is required to indicate
                the number of elements in each of the graph forming the packing. Original order is expected to be conserved.

        Returns
        -------
            out (torch.FloatTensor of size B x M): Aggregated features for the B graphs inside the input.

        """
        if self.pack_batch:
            if not nodes_per_graph:
                raise ValueError("Expect node_per_mol for packed graph")
            return torch.squeeze(torch.stack([self._pooling(mol_feat)
                                              for mol_feat in torch.split(h, nodes_per_graph, dim=1)], dim=0), dim=1)
        return torch.squeeze(self._pooling(h), dim=1)

    def forward(self, G, x, mask=None):
        G_size = self.G_size
        if not self.pack_batch and isinstance(G, (list, tuple)):
            G = torch.stack(G)
            x = torch.stack(x)  # .requires_grad_()

        if not isinstance(G, torch.Tensor) and self.pack_batch:
            G, h = pack_graph(G, x, False)
            G_size = h.shape[0]

        else:  # expect a batch here
            # ensure that batch dim is there
            xshape = x.shape[2] if x.dim() > 2 else x.shape[1]
            G = G.view(-1, G.shape[-2], G.shape[-1])
            h = x.view(-1, G.shape[1], xshape)
            G_size = h.shape[-2]

        # this could be important when we update the adjacency matrix after pooling
        #G_self = torch.diag_embed(torch.diagonal(G.permute(1, 2, 0))).to(G.device)
        G_self = torch.zeros_like(G)
        out = torch.matmul(G - G_self, h)
        out = (1 + self.eps) * h + out
    
        out = self.net(out.view(-1, self.in_size))
        out = out.view(-1, G_size, self.out_size)
        if mask is not None:
            out = out * mask.view(-1, G_size, 1).to(out.dtype)
        return G, out

class EdgeGraphLayer(nn.Module):
    def __init__(self, feat_dim, nedges, out_size=32, depth=2, pooling='max'):
        super(EdgeGraphLayer, self).__init__()
        self.feat_dim = feat_dim
        self.edge_dim = nedges
        self.out_size = out_size
        self.update_layers = nn.ModuleList()
        self.input_layer = nn.Linear(self.feat_dim, self.out_size)
        self.depth = depth
        for d in range(self.depth):
            self.update_layers.append(FCLayer(self.out_size + self.edge_dim, self.out_size))
        self.pooling = get_pooling(pooling)
        self.readout = FCLayer(self.out_size, self.out_size)

    @property
    def output_dim(self):
        return self.out_size

    def pack_graph(self, glist, nattr="h", eattr="he"):
        if isinstance(glist, dgl.BatchedDGLGraph):
            return glist
        return dgl.batch(glist, nattr, eattr)

    def pack_as_dgl(self, G, x, mask):
        glist = dgl_from_edge_matrix(G, x, mask=mask, full_mat=False)
        return self.pack_graph(glist)

    def _update_nodes(self, batch_G):
        h = batch_G.ndata['h']
        return {'hv': self.input_layer(h)}

    def _message(self, edges):
        return {"src_h": edges.src['hv'], "he": edges.data["he"]}

    def _reduce(self, nodes, updt=0):
        hw = nodes.mailbox['src_h']  # Batch, Deg, Feats
        he = nodes.mailbox['he'] # batch, deg, nedges
        out = torch.cat([hw, he], dim=-1) #(B, deg, D+D+E)
        b_size = out.size(0)
        msg_size = out.size(-1)
        out = self.update_layers[updt](out.view(-1, msg_size))
        return {"msg": self.pooling(out.view(b_size, -1, self.out_size))}

    def _apply(self, nodes):
        hv = nodes.data['msg'] + nodes.data['hv']
        return {'hv': hv}

    def _forward(self, batch_G):
        batch_G.ndata.update(self._update_nodes(batch_G))
        for updt in range(self.depth):
            batch_G.update_all(self._message, partial(self._reduce, updt=updt), self._apply)
        return self.readout(batch_G.ndata['hv'])

    def forward(self, G, x, mask=None):
        dgl_graph = self.pack_as_dgl(G, x, mask)
        out = self._forward(dgl_graph)
        adj_mat = adj_mat_from_edges(G)
        return adj_mat, out


# class EdgeGraphLayer(nn.Module):
#     def __init__(self, input_dim, nedges=1, output_dim=None, method="cat", basemodule=GINLayer, **module_params):
#         super(EdgeGraphLayer, self).__init__()
#         self.glayers = nn.ModuleList()
#         self.input_dim = input_dim
#         self._output_dim = output_dim or input_dim
#         for e in range(nedges):
#             self.glayers.append(basemodule(input_dim, self._output_dim, **module_params))
#         self.concat = ("cat" in method)
#         self.nedges = nedges

#     @property
#     def output_dim(self):
#         if self.concat:
#             return self._output_dim * self.nedges
#         return self._output_dim

#     def forward(self, G, x, mask=None):
#         # G is expected to be "B, N, N, E"
#         x_list = []
#         G_slice = torch.unbind(G, dim=-1)
#         for i, G_e in enumerate(G_slice):
#             _, x_e = self.glayers[i](G_e, x)
#             x_list.append(x_e.squeeze(0))
#         G_f = torch.stack(G_slice[:-1]).sum(dim=0)
#         if self.concat:
#             X_f = torch.cat(x_list, dim=-1)
#         else:
#             X_f = torch.stack(x_list, dim=0).sum(dim=0)

#         if mask is not None:
#             X_f = X_f * mask.view(X_f.shape[0], X_f.shape[1], 1).to(X_f.dtype)

#         return G_f, X_f 