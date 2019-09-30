import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from gnnpooling.utils.graph_utils import sample_sigmoid, mol2svg, restack, convert_to_grid, data2mol, convert_mol_to_smiles
from gnnpooling.utils.graph_utils import batch_full_to_triu, batch_triu_to_full, get_largest_fragment
from gnnpooling.utils.sparsegen import Sparsemax
from gnnpooling.models.gcn import GINLayer, EdgeGraphLayer
EPS = 1e-8

class AAE(nn.Module):
    """Discriminator network"""
    def __init__(self, atom_decoder, max_size, nedges, metrics={}, mse_loss=False, **kwargs):
        super(AAE, self).__init__()
        self.discriminator = None
        self.encoder = None
        self.decoder = None
        self.critic = None
        self.metrics = metrics
        self.sample_dim = -1
        self.atom_decoder = atom_decoder
        self.g_size = max_size
        self.nedges = nedges
        self.mse_loss = mse_loss
        self.n = 0

    def postprocess(self, inputs, temp=1, gumbel=False, hard=False, **kwargs):
        G, node_logits, *f = inputs
        n_edges = G.shape[-1]
        if gumbel: 
            G = F.gumbel_softmax(G.view(-1, self.nedges), tau=temp, hard=hard).view(G.size())
            x = F.gumbel_softmax(node_logits.view(-1, node_logits.size(-1)), tau=temp, hard=hard).view(node_logits.size())
        else:
            if hard:
                _, x_max = node_logits.max(dim=-1, keepdim=True)
                x =  torch.zeros_like(node_logits).scatter_(-1, x_max, 1)
                _, G_max = G.max(dim=-1, keepdim=True)
                G = torch.zeros_like(G).scatter_(-1, G_max, 1)
            else:
                x = F.softmax(node_logits.view(-1, node_logits.size(-1)), dim=-1).view(node_logits.size())
                G = F.softmax(G.view(-1, self.nedges), dim=-1).view(G.size()) #sample_sigmoid(G, hard=hard, gumbel=gumbel, **kwargs)
        G = batch_triu_to_full(G, self.g_size, self.nedges)
        f = [torch.sigmoid(feat) for feat in f]
        return (G, x, *f)

    def G_params(self):
        g_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.critic is not None:
            g_params += list(self.critic.parameters())
        return g_params

    def D_params(self):
        return list(self.discriminator.parameters())

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_encoder(self, encoder):
        self.encoder = encoder

    def set_critic(self, critic):
        self.critic = critic

    def set_decoder(self, decoder):
        self.decoder = decoder
        self.sample_dim = self.decoder.z_dim

    def forward_encoder(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def forward_decoder(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def forward_discriminator(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    def sample(self, batch_size):
        with torch.no_grad():
            return torch.randn(batch_size, self.sample_dim)

    def autoencoder_loss(self, real_data, rec_data, z, real_mols):
        edges, nodes, mask = real_data
        nodes_max = nodes.argmax(dim=-1)
        edges_triu = batch_full_to_triu(edges, self.g_size)
        #edges_rest = batch_triu_to_full(edges_triu, self.g_size, self.nedges)
        #assert torch.all(edges_rest==edges)

        if isinstance(edges, tuple):
            edges = torch.stack(edges)
        if isinstance(nodes, tuple):
            nodes_max = torch.stack(nodes_max)
        
        edges_hat, nodes_hat = rec_data
        #edges_hat_full = batch_triu_to_full(edges_hat, self.g_size, self.nedges)
        #print(edges_hat.view(-1).shape, edges_triu.argmax(dim=-1).view(-1).shape)
        edges_loss = F.cross_entropy(edges_hat.view(-1, edges_hat.size(-1)), edges_triu.argmax(dim=-1).long().view(-1), size_average=False) #
        node_loss = F.cross_entropy(nodes_hat.view(-1, nodes_hat.size(-1)), nodes_max.view(-1), size_average=False)
        recon_loss = (edges_loss + node_loss) / edges_hat.shape[0]

        if self.mse_loss:
            edges_hat, nodes_hat, *add_feat_hat = self.postprocess(rec_data, gumbel=True, hard=False)
            real_encoding = compute_gconv_loss(edges, nodes)
            rec_encoding = compute_gconv_loss(edges_hat, nodes_hat)
            recon_loss += F.mse_loss(rec_encoding, real_encoding, reduction="mean")

        if self.critic is not None:
            value_logit_real, _ = self.critic(z)
            reward = self.reward(real_mols).to(value_logit_real.device)
            recon_loss += F.mse_loss(value_logit_real, reward)
        return recon_loss

    def oneside_discriminator_loss(self, out, truth=False):
        out, _ = out
        if truth:
            y = torch.ones_like(out).requires_grad_(False)
        else:
            y = torch.zeros_like(out).requires_grad_(False)
        return F.binary_cross_entropy(out, y)

    def discriminator_loss(self, true_out, fake_out): 
        loss = self.oneside_discriminator_loss(true_out, truth=True) + self.oneside_discriminator_loss(fake_out)
        return loss

    def reward(self, mols):
        rr = 1.
        for m, v in self.metrics.items():
            r = v(mols)
            rr *= r
        rr = torch.from_numpy(rr).float()
        return rr.view(-1, 1)


    def log(self, mols, rec_mols, ori_graph, writer, step=1):
        #images = convert_to_grid(mols)
        rec_images = convert_to_grid(rec_mols)
        graph_images = convert_to_grid(ori_graph)
        if not (writer is None):
            #writer.add_image('orimols', images, step)
            writer.add_image('genmols', rec_images, step)
            writer.add_image('origraph', graph_images, step)

    def as_mol(self, data, largest_fragment=False):
        mol = data2mol(data, self.atom_decoder)
        if largest_fragment:
            mol = [get_largest_fragment(m) for m in mol]
        return mol

    def mols_to_smiles(self, mols):
        return convert_mol_to_smiles(mols)


def compute_gconv_loss(adj, x, *x_add):
    if x_add:
        x = torch.cat([x]+list(x_add), dim=-1)
    x = torch.stack([x for _ in range(adj.size(-1))], dim=1)
    return torch.einsum('bijk,bkjl->bil', (adj, x)).sum(dim=-2)