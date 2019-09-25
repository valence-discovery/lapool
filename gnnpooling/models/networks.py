import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gnnpooling.models.gcn import EdgeGraphLayer, GINLayer, GCNLayer
from gnnpooling.models.layers import FCLayer, AggLayer, get_activation, get_pooling
from gnnpooling.pooler import get_graph_coarsener
from gnnpooling.utils.nninspect.gradinspect import GradientInspector
from gnnpooling.utils.nninspect.poolinspector import PoolingInspector

GLayer = GINLayer


class Encoder(nn.Module):
    """Encoder for AAE (just latent rep, no VAE)."""

    def __init__(self, input_dim, out_dim, feat_dim=0, nedges=1, gather_dim=72, conv_dims=[64, 64],
                 conv_dims_after=[128, 128], linear_dim=[32], dropout=0.2, gather="agg", pool_arch={},
                 pool_loss=False, **kwargs):
        super(Encoder, self).__init__()
        GraphLayer = kwargs.pop("GraphConv", GLayer)  # using GIN convolution when none is specified
        activation = kwargs.pop("activation", nn.LeakyReLU())  # Leaky relu
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.feat_dim = feat_dim  # when additional features are available for the node
        self.pool_loss = pool_loss
        self.nedges = nedges
        self.conv_layers1 = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        input_dim += feat_dim
        if nedges > 1:
            self.edge_layer = EdgeGraphLayer(input_dim, nedges, method='cat', b_norm=False, dropout=0.0,
                                         activation=activation)
            input_dim = self.edge_layer.output_dim + self.feat_dim
        for cdim in conv_dims:
            self.conv_layers1.append(
                GraphLayer(input_dim, kernel_size=cdim, b_norm=False, dropout=dropout, activation=activation))
            input_dim = cdim

        self.pool_layer = get_graph_coarsener(**pool_arch)
        self.pool_arch = pool_arch.get("arch", "")
        if self.pool_layer is not None:
            self.pool_layer = self.pool_layer(input_dim)
            if self.pool_arch == "laplacian":
                input_dim = self.pool_layer.cluster_dim

        for cdim in conv_dims_after:
            self.conv_layers2.append(
                GraphLayer(input_dim, kernel_size=cdim, b_norm=False, dropout=dropout, activation=activation))
            input_dim = cdim

        if gather == "agg":
            self.agg_layer = AggLayer(input_dim, gather_dim, dropout)
        elif gather in ["attn", "gated"]:
            self.agg_layer = get_pooling("attn", input_dim=input_dim, output_dim=gather_dim, dropout=dropout)
        else:
            gather_dim = input_dim
            self.agg_layer = get_pooling(gather)
        # multi dense layer
        input_dim = gather_dim

        self.linear_layers = nn.ModuleList()
        for ldim in linear_dim:
            self.linear_layers.append(FCLayer(input_dim, ldim, activation=activation, **kwargs))
            input_dim = ldim

        self.output_layer = nn.Linear(input_dim, out_dim)
        self.pool_inspect = None
        self.pooling_loss = []

    def set_inspector(self, inspect, *args, show=True, freq=8, save_video=False, **kwargs):
        if inspect and self.pool_layer is not None:
            self.pool_inspect = PoolingInspector(self.pool_arch, *args, show=show, freq=freq, save_video=save_video,
                                                 **kwargs)

    def _forward(self, x, mols=None):
        # Prepare the inputs
        self.pooling_loss = []
        hidden = None
        # add additional feat dim if required
        try:
            adj_list, feats, hidden = x
        except:
            adj_list, feats = x
        # n_per_mols = [g.shape[0] for g in adj_list]
        h_result = []
        for idx, G in enumerate(adj_list):
            # add non-mandatory extra features
            h = feats[idx]
            if mols:
                mol = mols[idx]
            if self.nedges > 1:
                G, h = self.edge_layer(G, h)
            else:
                G = G.squeeze(-1)
            if hidden and self.feat_dim > 0:
                h = torch.cat([h, hidden[idx]], dim=-1)
            # Run graph convolution
            for clayer in self.conv_layers1:
                h_in = h
                G, h = clayer(G, h)

            if self.pool_layer is not None:  # run coarsening model
                old_G = G
                sup_args = {}
                if "diff" in self.pool_arch:
                    sup_args = {"x": h_in}
                if self.pool_loss == True:
                    G, h, side_loss = self.pool_layer(G, h, return_loss=True, **sup_args)
                    self.pooling_loss += [side_loss]
                else:
                    G, h = self.pool_layer(G, h, **sup_args)

                if self.pool_inspect is not None and mol:
                    self.pool_inspect(self.pool_layer, old_G, G, mol, training=self.training)

            for clayer in self.conv_layers2:
                G, h = clayer(G, h)

            h = self.agg_layer(h)  # aggregating at graph level

            h_result.append(h)
        
        h_batch = torch.cat(h_result, dim=0)
        for layer in self.linear_layers:
            h_batch = layer(h_batch)
        return h_batch

    def _sideloss(self):
        if len(self.pooling_loss) > 0:
            return torch.mean(torch.stack(self.pooling_loss))
        return 0

    def forward(self, x, mols=None):
        pre_output = self._forward(x, mols=mols)
        return self.output_layer(pre_output)


class Discriminator(Encoder):
    """Discriminator network."""

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.out_dim = 1
        input_dim = self.linear_layers[-1].output_dim

        self.output_layer = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, x, mols=None):
        pre_output = self._forward(x, mols=mols)
        return self.output_layer(pre_output), pre_output


class MLPdiscriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, z_dim, layers_dim=[256, 64, 64, 32], dropout=0., **kwargs):
        super(MLPdiscriminator, self).__init__()
        self.out_dim = 1
        layers = []
        for in_dim, out_dim in zip([z_dim] + layers_dim[:-1], layers_dim):
            layers.append(FCLayer(in_dim, out_dim, dropout=dropout, **kwargs))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid())

    def forward(self, x):
        pre_output = self.layers(x)
        return self.output_layer(pre_output), pre_output


class Decoder(nn.Module):
    """Decoder network."""

    def __init__(self, z_dim, node_feat_dim, nedges=1, max_vertex=50, layers_dim=[256, 512], nodes_dim=[], graph_dim=[],
                 dropout=0., activation="relu", other_feat_dim=0, **kwargs):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.graph_size = max_vertex
        self.node_feat_dim = node_feat_dim
        self.feat_dim = other_feat_dim
        self.nedges = nedges

        layers = []
        for in_dim, out_dim in zip([z_dim] + layers_dim[:-1], layers_dim):
            layers.append(FCLayer(in_dim, out_dim, activation=activation, b_norm=True))
        self.layers = nn.Sequential(*layers)

        adj_layers = []
        in_dim = out_dim
        for adj_dim in graph_dim:
            adj_layers.append(FCLayer(in_dim, adj_dim, dropout=dropout, activation=activation, b_norm=True, bias=False))
            in_dim = adj_dim

        nnodes = self.graph_size * (self.graph_size - 1) // 2  # correspond to upper triangular
        adj_layers += [nn.Linear(in_dim, nnodes * self.nedges)]
        self.adj_layer = nn.Sequential(*adj_layers)

        node_layers = []
        in_dim = layers_dim[-1]
        for n_dim in nodes_dim:
            node_layers.append(FCLayer(in_dim, n_dim, dropout=dropout, activation=activation, b_norm=True, bias=False))
            in_dim = n_dim

        node_layers += [nn.Linear(in_dim, self.graph_size * self.node_feat_dim)]
        self.nodes_layer = nn.Sequential(*node_layers)

        if self.feat_dim:
            self.feat_layer = nn.Linear(layers_dim[-1], self.graph_size * self.feat_dim)
        else:
            self.feat_layer = None
        ind = np.ravel_multi_index(np.triu_indices(self.graph_size, 1), (self.graph_size, self.graph_size))
        self.upper_tree = torch.zeros(self.graph_size ** 2).index_fill(0, torch.from_numpy(ind),
                                                                       1).contiguous().unsqueeze(-1).expand(
            self.graph_size ** 2, self.nedges).byte()

    @property
    def output_dim(self):
        return (self.node_feat_dim, self.feat_dim, self.graph_size, self.nedges)

    def forward(self, x):
        output = self.layers(x)
        b_size = output.shape[0]
        if self.upper_tree.device != output.device:
            self.upper_tree = self.upper_tree.to(output.device)
        adj_logits_triu = self.adj_layer(output)
        adj_logits = adj_logits_triu.new_zeros((b_size, self.graph_size ** 2, self.nedges))
        adj_logits = adj_logits.masked_scatter(self.upper_tree, adj_logits_triu.view(b_size, -1, self.nedges)).view(-1,
                                                                                                                    self.graph_size,
                                                                                                                    self.graph_size,
                                                                                                                    self.nedges)
        adj_logits = adj_logits + adj_logits.transpose(-2, -3)

        # not transform to matrix     
        nodes_logits = self.nodes_layer(output)
        nodes_logits = nodes_logits.view(-1, self.graph_size, self.node_feat_dim)
        if self.feat_dim:
            feats_logits = self.feat_layer(output)
            feats_logits = feats_logits.view(-1, self.graph_size, self.feat_dim)
            return adj_logits, nodes_logits, feats_logits
        return adj_logits, nodes_logits
