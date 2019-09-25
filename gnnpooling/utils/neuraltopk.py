import torch
import torch.nn as nn
import torch.nn.functional as F

def log1mexp(x, expm1_guard=1e-8):
    # See https://cran.r-project.org/package=Rmpfr/.../log1mexp-note.pdf
    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())
    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[1 - t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1+expm1_guard).log() # limits magnitude of gradient
    y[1 - t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y


class NeuralTopK(nn.Module):
    r"""
    Relaxed version of the top-k selection
    based on https://arxiv.org/pdf/1810.12575.pdf
    """
    def __init__(self, k, temp=None):
        r"""
        :param k: Number of sample to choose
        :param temp: temperature options, will be learned if 0
        """
        super(NeuralTopK, self).__init__()
        self.temp = temp
        if self.temp is None:
            self.temp = nn.Parameter(torch.FloatTensor(1).fill_(0.0))
        self.k = k

    def forward(self, logits, log_temp=None):
        samples_arr = []
        logits = logits / (self.temp + 1) ## adding a fixed bias to avoid division by zero ?
        # logits is expected to be n*1
        for r in range(self.k):
            # Eqs. 8 and 10
            weights = F.log_softmax(logits, dim=0)
            weights_exp = weights.exp()
            samples_arr.append(weights_exp)
            logits = logits + log1mexp(weights.view(*logits.shape))

        W = torch.stack(samples_arr)
        return W