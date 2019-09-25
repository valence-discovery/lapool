import torch
import numpy as np

def cosine_attn(x1, x2, eps=1e-8):
    """Compute attention using cosine similarity"""
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = x2.norm(p=2, dim=-1, keepdim=True)
    return torch.matmul(x1, x2.transpose(-2, -1)) / (w1 * w2.transpose(-2, -1)).clamp(min=eps)

def dot_attn(x1, x2, **kwargs):
    attn = torch.matmul(x1, x2.transpose(-2, -1)) # B, N, M 
    return attn / np.sqrt(x1.shape[-1])

def upsample_to(vec, d):
    """Convert a N,F vector to N*d, F vector (for concatenation purpose
    Not sure if this is smarter than just using for loop. But fight me!"""
    vec = vec.view(-1, vec.shape[-1])
    return vec.unsqueeze(0).expand(d, vec.shape[0], vec.shape[1]).contiguous().view(-1, vec.shape[-1])

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones_like(y)
    dydx = torch.autograd.grad(outputs=y,
                                 inputs=x,
                                 grad_outputs=weight,
                                 retain_graph=True,
                                 create_graph=True,
                                 only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    return ((dydx.norm(2, dim=1) - 1) ** 2).mean()


def is_tensor(dtype):
    return isinstance(dtype, torch.dtype) or (dtype == torch.Tensor)


def is_numpy(dtype):
    is_torch = is_tensor(dtype)
    is_num = dtype in (int, float, complex)
    if hasattr(dtype, '__module__'):
        is_numpy = dtype.__module__ == 'numpy'
    else:
        is_numpy = False

    return (is_num or is_numpy) and not is_torch


def one_of_k_encoding(val, num_classes, dtype=int):
    encoding = np.zeros(len(num_classes) + 1, dtype=dtype)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(num_classes):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found
        encoding[-1] = 1
    return encoding


def to_tensor(x, gpu=True, dtype=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.Tensor(x)
    if dtype is not None:
        x = x.type(dtype)
    if torch.cuda.is_available() and gpu:
        x = x.cuda()
    return x


def to_sparse(x, dtype=None):
    r"""
    Converts dense tensor x to sparse format
    """
    if dtype is not None:
        x = x.type(dtype)

    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def batch_index_select(input, dim, index):    
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)