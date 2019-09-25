import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from gnnpooling.utils.graph_utils import sample_sigmoid, mol2svg, restack, convert_to_grid, data2mol
from gnnpooling.utils.sparsegen import Sparsemax
from gnnpooling.models.gcn import GINLayer, EdgeGraphLayer

EPS = 1e-8

class AAE(nn.Module):
    """Discriminator network"""
    def __init__(self, autoreg=False, metrics={}, **kwargs):
        super(AAE, self).__init__()
        self.discriminator = None
        self.encoder = None
        self.decoder = None
        self.critic = None
        self.x_dim = self.g_dim = 0
        self.metrics = metrics
        self.sample_dim = -1
        self.autoreg = autoreg
        self.softmax = Sparsemax()

    def postprocess(self, inputs, temperature=1., hard=False, **kwargs):
        G, node_logits, *f = inputs
        if self.autoreg:
            return inputs
        x = self.softmax(node_logits)# / temperature, dim=-1)
        if hard:
            _, x_max = x.max(dim=-1, keepdim=True)
            x =  torch.zeros_like(x).scatter_(-1, x_max, 1)

        G = sample_sigmoid(G, hard=hard, gumbel=False, **kwargs)
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

    def autoencoder_loss(self, z, real_data, real_mols, rec_data, rec_mols):
        if not self.autoreg:
            # might experiment with L1 loss here: F.l1_loss, since it could ensure more sparsity
            edges, nodes, *add_feat = real_data
            if isinstance(edges, tuple):
                edges = torch.stack(edges)
            if isinstance(nodes, tuple):
                nodes = torch.stack(nodes)
            add_feat = [torch.stack(x) for x in add_feat]
            edges_hat, nodes_hat, *add_feat_hat = rec_data
            if True:
                _, nodes_max = nodes.max(dim=-1)
                recon_loss = F.binary_cross_entropy_with_logits(edges_hat, edges) + F.cross_entropy(nodes_hat.view(-1, nodes_hat.size(-1)), nodes_max.view(-1))
                for d_i, d_j in zip(add_feat, add_feat_hat):
                    recon_loss += F.binary_cross_entropy_with_logits(d_j, d_i)
            else:
                edges_hat, nodes_hat, *add_feat_hat = self.postprocess(rec_data, hard=False)
                real_encoding = compute_gconv_loss(edges, nodes, *add_feat)
                rec_encoding = compute_gconv_loss(edges_hat, nodes_hat, *add_feat_hat)
                recon_loss += F.mse_loss(rec_encoding, real_encoding, reduction="mean")
        else:
            raise ValueError("autoreg not implemented yet")

        if self.critic is not None:
            value_logit_real, _ = self.critic(z)
            reward = self.reward(real_mols).to(value_logit_real.device)
            recon_loss += F.mse_loss(value_logit_real, reward)
        return recon_loss

    def oneside_discriminator_loss(self, out, truth=False):
        out, _ = out
        if truth:
            y = torch.ones_like(out)
        else:
            y = torch.zeros_like(out)
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


    def log(self, mols, rec_mols, writer, step=1):
        ori_images = convert_to_grid(mols)
        rec_images = convert_to_grid(rec_mols)
        if not (writer is None):
            writer.add_image('orimols', ori_images, step)
            writer.add_image('recmols', rec_images, step)


def compute_gconv_loss(adj, x, *x_add):
    if x_add:
        x = torch.cat([x]+list(x_add), dim=-1)
    x = torch.stack([x for _ in range(adj.size(-1))], dim=1)
    return torch.einsum('bijk,bkjl->bil', (adj, x)).sum(dim=-2)