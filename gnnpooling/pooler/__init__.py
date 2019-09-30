from .diffpool import DiffPool
from .topkpool import TopKPool
from .clusterpool import ClusterPool
from .batchlapool import LaPool

from functools import partial
from gnnpooling.models.layers import FCLayer, GlobalSoftAttention, GlobalMaxPool1d, GlobalSumPool1d, GlobalGatedPool, DeepSetEncoder, GlobalAvgPool1d

def get_hpool(**params):
    arch = params.pop("arch", None)
    if arch is None:
        return None, None
    if arch == "diff":
        pool = partial(DiffPool, **params)
    elif arch == "topk":
        pool = partial(TopKPool, **params)
    elif arch == "cluster":
        pool = partial(ClusterPool, algo="hierachical", gweight=0.5, **params)
    elif arch == "laplacian":
        pool = partial(LaPool, **params)
    return pool, arch


def get_gpool(input_dim=None, **params):
    arch = params.pop("arch", None)
    if arch == 'gated':
        return GlobalGatedPool(input_dim, input_dim, **params)  # dropout=0.1
    elif arch == 'deepset':
        return DeepSetEncoder(input_dim, input_dim, **params)
    elif arch == 'attn':
        return GlobalSoftAttention(input_dim, input_dim, **params)
    elif arch == 'avg':
        return GlobalAvgPool1d()
    elif arch == 'max':
        return GlobalMaxPool1d()
    return GlobalSumPool1d()
