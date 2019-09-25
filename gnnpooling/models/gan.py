import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnnpooling.utils.graph_utils import sample_sigmoid, mol2svg, convert_to_grid
from gnnpooling.utils.tensor_utils import gradient_penalty

class GAN(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, gp, sampler="soft_gumbel", autoreg=False, metrics={}):
        super(GAN, self).__init__()
        self.discriminator = None
        self.generator = None
        self.critic = None
        self.gp = gp
        self.feat_sampler = sampler
        self.metrics = metrics
        self.sample_dim = -1
        self.autoreg = autoreg

    def postprocess(self, inputs, temperature=1., hard=False, **kwargs):
        G, node_logits, *f = inputs
        if self.autoreg:
            return (G, *x)

        method = kwargs.pop("method", self.feat_sampler)
        if method == 'hard_gumbel' or hard:
            x = F.gumbel_softmax(node_logits.contiguous().view(-1, node_logits.size(-1)), tau=temperature, hard=True).view(node_logits.size())
        elif method == 'soft_gumbel':
            x = F.gumbel_softmax(node_logits.contiguous().view(-1, node_logits.size(-1)), tau=temperature, hard=False).view(node_logits.size())
        else:
            x = F.softmax(node_logits / temperature, -1)

        G = sample_sigmoid(G, temperature, hard=hard, **kwargs)
        f = [torch.sigmoid(feat) for feat in f]
        return (G, x, *f)

    def G_params(self):
        g_params = list(self.generator.parameters())
        if self.critic is not None:
            g_params += list(self.critic.parameters())
        return g_params

    def D_params(self):
        return list(self.discriminator.parameters())
        
    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_generator(self, generator):
        self.generator = generator
        self.sample_dim = self.generator.z_dim

    def set_critic(self, critic):
        self.critic = critic

    def sample(self, batch_size):
        with torch.no_grad():
            return torch.randn(batch_size, self.sample_dim)

    def forward_generator(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def forward_discriminator(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    def adversarial_loss(self, logits_real, logits_fake, real_data, fake_data):
        logits_real, f_real = logits_real
        logits_fake, f_fake = logits_fake
        loss = -torch.mean(logits_real) + torch.mean(logits_fake)
        try:
            sd_loss = self.discriminator._sideloss()
            loss += sd_loss
        except:
            pass

        if self.gp:
            loss += self.gp*self.penalty(real_data, fake_data)
        return loss

    def generator_loss(self, logits_fake, real_data, real_mols, fake_data, fake_mols):
        g_loss_value = 0
        logits_fake, _ = logits_fake
        if self.critic is not None:
            # Real Reward
            rewardR = self.reward(real_mols).to(logits_fake.device)
            # Fake Reward         mols = data2mol(data)
            rewardF = self.reward(fake_mols).to(logits_fake.device)
            # Value loss
            value_logit_real, _ = self.critic(real_data)
            value_logit_fake, _ = self.critic(fake_data)
            g_loss_value =  F.mse_loss(value_logit_real, rewardR) + F.mse_loss(value_logit_fake, rewardF) 
        g_loss_fake = -torch.mean(logits_fake)
        return g_loss_fake + g_loss_value

    def penalty(self, real_data, fake_data):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        if not isinstance(real_data, (tuple,list)):
            real_data = [real_data]

        if not isinstance(fake_data, (tuple,list)):
            fake_data = [fake_data]

        if len(real_data) > len(fake_data):
            real_data = real_data[:-1]
            
        b_size = fake_data[0].size(0)
        alpha = torch.rand(b_size,1,1,1).to(fake_data[0].device)
        sample_list = []
        for r_sample, f_sample in zip(real_data, fake_data):
            r_sample = restack(r_sample)
            sample_list.append((alpha * r_sample + ((1 - alpha) * f_sample)).requires_grad_(True))

        sample_list = tuple(sample_list)
        grads = self.forward_discriminator(sample_list)
        if not isinstance(grads, (tuple, list)):
            grads = [grads]
    
        return sum(gradient_penalty(grads[k], sample_list) for k in range(len(grads)))


    def reward(self, mols):
        rr = 1.
        for m, v in self.metrics.items():
            r = v(mols)
            rr *= r
        rr = torch.from_numpy(rr).float()
        return rr.view(-1, 1)

    def log(self, mols, writer, step=1):
        images = convert_to_grid(mols)
        if not (writer is None or images is None):
            writer.add_image('mols', images, step)
