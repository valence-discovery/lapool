"""Sparsegen activation function.

Pytorch implementation of Sparsegen function from:

https://arxiv.org/pdf/1810.11975.pdf

The source code of this implementation is based on the SparseMax layer (in lua) of 
https://github.com/gokceneraslan/SparseMax.torch/

The sparsemax is in fact a special class of the Sparsegen layer. 
See: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification" : http://arxiv.org/abs/1602.02068

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparsegen(nn.Module):
    """Sparsegen function."""

    def __init__(self, dim=None, net=nn.Sequential(), sigma=0):
        """Initialize Sparsegen activation. By default, it performs 
        SparseMax

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsegen, self).__init__()

        self.dim = -1 if dim is None else dim
        self.net = net
        self.sigma = sigma

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = self.net(input)
        input = input.view(-1, input.size(self.dim))
        
        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0]

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        trange = torch.arange(start=1, end=number_of_logits+1, device=input.device).float().view(1, -1)
        trange = trange.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 - self.sigma + trange * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * trange, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1 + self.sigma) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), (input - taus)/(1-self.sigma))
        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum_all = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum_all.expand_as(grad_output))
        return self.grad_input


class Sparsemax(Sparsegen):
    def __init__(self, dim=None):
        super().__init__(dim=None)
