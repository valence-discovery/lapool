import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gnnpooling.models.gcn import EdgeGraphLayer, GINLayer, GCNLayer
from gnnpooling.models.layers import FCLayer, get_activation
from gnnpooling.pooler import get_gpool, get_hpool


GraphLayer = GINLayer

class Encoder(nn.Module):
    """Encoder for AAE (just latent rep, no VAE)."""

    def __init__(self, input_dim, out_dim, nedges, **config):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.nedges = nedges
        self.conv_layers1 = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        self.pool_loss = 0
        self._embedding_layer = None
        if config.get('embed_dim'):
            self._embedding_layer = nn.Embedding(self.input_dim, config['embed_dim'])
            input_dim = config.get('embed_dim', self.input_dim)
        if nedges > 1:
            self.edge_layer = EdgeGraphLayer(self.input_dim, nedges, **config.get("edge_layer", {}))
            input_dim = self.edge_layer.output_dim
        
        conv_before_dims = config['conv_before']
        for conf in conv_before_dims:
            self.conv_layers1.append(GraphLayer(input_dim, **conf))
            input_dim = conf['kernel_size']
        
        self.hpool, self.hpool_arch = get_hpool(**config["hpool"])
        if self.hpool_arch is not None:
            self.hpool = self.hpool(input_dim=input_dim)
        if self.hpool_arch == 'laplacian':
            input_dim = self.hpool.cluster_dim

        conv_after_dims = config['conv_after']
        for conf in conv_after_dims:
            self.conv_layers2.append(GraphLayer(input_dim, **conf))
            input_dim = conf['kernel_size']

        self.gpooler = get_gpool(input_dim=input_dim, **config["gpool"])
        self.linear_layers = nn.ModuleList()
        for conf in (config.get("fc_layers") or []):
            self.linear_layers.append(FCLayer(input_dim, **conf))
            input_dim = conf["out_size"]

        self.output_layer = nn.Linear(input_dim, self.out_dim)

    def embedding_layer(self, x):
        if self._embedding_layer is not None:
            return self._embedding_layer(torch.argmax(x, dim=-1))
        return x

    def _forward(self, G, x, mask=None):
        # Prepare the inputs
        self.pooling_loss = 0

        x = self.embedding_layer(x)
        if self.nedges > 1:
            G, h = self.edge_layer(G, x, mask=mask)
        else:
            G = G.squeeze(-1)
            h = x
        
        for clayer in self.conv_layers1:
            x = h
            G, h = clayer(G, h, mask=mask)
        
        if not self.hpool_arch:
            pass

        elif self.hpool_arch == 'diff':
            G, h, self.pooling_loss = self.hpool(G, z=h, x=x, return_loss=True, mask=mask)
        else:
            G, h, self.pooling_loss = self.hpool(G, h, return_loss=True, mask=mask)

        for clayer in self.conv_layers2:
            G, h = clayer(G, h)
        h = self.gpooler(h)
        for fc_layer in self.linear_layers:
            h = fc_layer(h)        
        return h

    def _sideloss(self):
        return self.pooling_loss

    def forward(self, G, x, mols=None, mask=None):
        pre_output = self._forward(G, x, mask=mask)
        return self.output_layer(pre_output), self._sideloss()


class GANDiscriminator(Encoder):
    """Discriminator network."""

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.out_dim = 1
        input_dim = self.linear_layers[-1].output_dim
        self.output_layer = nn.Sequential(nn.Linear(input_dim, self.out_dim), nn.Sigmoid())

    def forward(self, x, mols=None):
        pre_output = self._forward(x, mols=mols)
        return self.output_layer(pre_output), pre_output


class MLPdiscriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, z_dim, layers_dim=[64, 64, 32], dropout=0., **kwargs):
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

    def __init__(self, z_dim, atom_dim, max_vertex, nedges=1, **config):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.graph_size = max_vertex
        self.node_feat_dim = atom_dim
        self.nedges = nedges

        layers = []
        in_dim = z_dim
        for conf in config['layers']:
            layers.append(FCLayer(in_dim, **conf))
            in_dim = conf['out_size']
        self.layers = nn.Sequential(*layers)
        
        adj_layers = []
        adj_in_dim = in_dim
        for gconf in config['graph_layers']:
            adj_layers.append(FCLayer(adj_in_dim,  **gconf))
            adj_in_dim = gconf['out_size']
        nnodes = self.graph_size * (self.graph_size - 1) // 2  # correspond to upper triangular
        adj_layers.append(nn.Linear(adj_in_dim, nnodes * self.nedges))
        self.adj_layers = nn.Sequential(*adj_layers)

        node_layers = []
        node_in_dim = in_dim
        for nconf in config['node_layers']:
            node_layers.append(FCLayer(node_in_dim,**nconf))
            node_in_dim = nconf['out_size']
        node_layers.append(nn.Linear(node_in_dim, self.graph_size * self.node_feat_dim))
        self.node_layers = nn.Sequential(*node_layers)

    @property
    def output_dim(self):
        return (self.node_feat_dim, self.graph_size, self.nedges)

    def forward(self, x):
        output = self.layers(x)
        b_size = output.shape[0]
        adj_logits = self.adj_layers(output)
        adj_logits = adj_logits.view(b_size, -1, self.nedges)
        nodes_logits = self.node_layers(output)
        nodes_logits = nodes_logits.view(b_size, self.graph_size, self.node_feat_dim)
        return adj_logits, nodes_logits

