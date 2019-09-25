import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh',
                            'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}

class GlobalMaxPool1d(nn.Module):
    r"""
    Global max pooling of a Tensor over one dimension
    """

    def __init__(self, dim=1):
        super(GlobalMaxPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class GlobalAvgPool1d(nn.Module):
    r"""
    Global Average pooling of a Tensor over one dimension
    """

    def __init__(self, dim=1):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class GlobalSumPool1d(nn.Module):
    r"""
    Global Sum pooling of a Tensor over one dimension
    """
    def __init__(self, dim=1):
        super(GlobalSumPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class GlobalSoftAttention(nn.Module):
    r"""
    Global soft attention layer for computing the output vector of a graph convolution.
    It's akin to doing a weighted sum at the end
    """

    def __init__(self, input_dim, output_dim=None, use_sigmoid=False, **kwargs):
        super(GlobalSoftAttention, self).__init__()
        # Setting from the paper
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not self.output_dim:
            self.output_dim == self.input_dim

        # Embed graphs
        sig = nn.Sigmoid(-2) if use_sigmoid else nn.Softmax(-2)
        self.node_gating = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            sig
        )
        self.pooling = get_pooling("sum", dim=-2)
        self.graph_pooling = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros(1, self.output_dim)
        else:
            x =  torch.mul(self.node_gating(x), self.graph_pooling(x))
            return self.pooling(x)


class GlobalGatedPool(nn.Module):
    r"""
    Gated global pooling layer for computing the output vector of a graph convolution.
    """
    def __init__(self, input_dim, output_dim, dim=-2, dropout=0., **kwargs):
        super(GlobalGatedPool, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dim = dim
        self.sigmoid_linear = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i, j), self.dim)  # on the node dimension
        output = self.dropout(output)
        return output


class UnitNormLayer(nn.Module):
    r"""
    Normalization layer. Performs the following operation: x = x / ||x||

    """

    def __init__(self, dim=1):
        super(UnitNormLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Unit-normalizes input x

        Arguments
        ----------
            x: torch.FloatTensor of size N*M
                Batch of N input vectors of size M.

        Returns
        -------
            x: torch.FloatTensor of size N*M
                Batch of normalized vectors along the dimension 1.
        """
        return F.normalize(x, dim=self.dim)


class ResidualBlock(nn.Module):
    r"""Residual Block maker
    Let :math:`f` be a module, the residual block acts as a module g such as :math:`g(x) = \text{ReLU}(x + f(x))`

    Arguments
    ----------
        base_module: torch.nn.Module
            The module that will be made residual
        resample: torch.nn.Module, optional
            A down/up sampling module, which is needed
            when the output of the base_module doesn't lie in the same space as its input.
            (Default value = None)
        auto_sample: bool, optional
            Whether to force resampling when the input and output
            dimension of the base_module do not match, and no resampling module was provided.
            By default, the `torch.nn.functional.interpolate` function will be used.
            (Default value = False)
        activation: str or callable
            activation function to use for the residual block
            (Default value = 'relu')
        kwargs: named parameters for the `torch.nn.functional.interpolate` function

    Attributes
    ----------
        base_module: torch.nn.Module
            The module that will be made residual
        resample: torch.nn.Module
            the resampling module
        interpolate: bool
            specifies if resampling should be enforced.

    """
    def __init__(self, base_module, resample=None, auto_sample=False, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__()
        self.base_module = base_module
        self.resample = resample
        self.interpolate = False
        self.activation = get_activation(activation)
        if resample is None and auto_sample:
            self.resample = partial(nn.functional.interpolate, **kwargs)
            self.interpolate = True

    def forward(self, x):
        residual = x
        indim = residual.shape[-1]
        out = self.base_module(x)
        outdim = out.shape[-1]
        if self.resample is not None and not self.interpolate:
            residual = self.resample(x)
        elif self.interpolate and outdim != indim:
            residual = self.resample(x, size=outdim)

        out += residual
        if self.activation:
            out = self.activation(out)
        return out


def get_activation(activation):
    if activation and callable(activation):
        return activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) > 0 and isinstance(activation[0], str), \
        'Unhandled activation function'
    if activation[0].lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation[0]]()


def get_pooling(pooling, **kwargs):
    if pooling and callable(pooling):
        return pooling
    # there is a reason for this to not be outside
    POOLING_MAP = {"max": GlobalMaxPool1d, "avg": GlobalAvgPool1d,
                   "sum": GlobalSumPool1d, "mean": GlobalAvgPool1d, 'attn': GlobalSoftAttention, 'attention': GlobalSoftAttention, 'gated': GlobalGatedPool}
    return POOLING_MAP[pooling.lower()](**kwargs)


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer

    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    @property
    def output_dim(self):
        return self.out_size

    def reset_parameters(self, init_fn=None):        
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            h = self.b_norm(h)
        return h


    def clone(self):
        for key in ['__class__', 'self']:
            if key in self.__params:
                del self.__params[key]

        model = self.__class__(**self.__params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model


class AggLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., activation="tanh"):
        super(AggLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.sigmoid_linear = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        if activation is not None:
            self.activation = get_activation(activation)
        else:
            self.activation = None

    def forward(self, input):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i, j), -2)  # on the node dimension
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)
        return output



class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=0, num_layers=0, functions='meanstd'):
        super(DeepSetEncoder, self).__init__()
        assert functions in ['meanstd', 'stdmean', 'maxsum', 'summax']
        layers = []
        in_dim, out_dim = input_dim, hidden_dim
        for i in range(1, num_layers + 1):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim
        self.phi_net = nn.Sequential(*layers)

        layers = []
        in_dim, out_dim = hidden_dim * 2, hidden_dim
        for i in range(1, num_layers + 1):
            layers.append(nn.Linear(in_dim, out_dim))
            # if i != num_layers:
            layers.append(nn.ELU())
            in_dim, out_dim = out_dim, out_dim
        self.rho_net = nn.Sequential(*layers)
        self.functions = functions
        self._output_dim = 2 * input_dim if num_layers <= 0 else hidden_dim

    def _forward(self, x):
        phis_x = self.phi_net(x)
        if self.functions in ['meanstd', 'stdmean']:
            x1 = torch.mean(phis_x, dim=0, keepdim=True)
            x2 = torch.sqrt(torch.var(phis_x, dim=0, keepdim=True) + torch.FloatTensor([1e-8]).to(x.device))
        else:
            x1 = torch.sum(phis_x, dim=0, keepdim=True)
            x2, _ = torch.max(phis_x, dim=0, keepdim=True)
        z = torch.cat([x1, x2], dim=1)
        res = self.rho_net(z).squeeze(0)
        return res

    def forward(self, x):
        return torch.stack([self._forward(x_i) for x_i in x], dim=0)

    @property
    def output_dim(self):
        return self._output_dim


class Set2Set(torch.nn.Module):
    r"""
    Set2Set global pooling operator from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper. This pooling layer performs the following operation

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Arguments
    ---------
        input_dim: int
            Size of each input sample.
        hidden_dim: int, optional
            the dim of set representation which corresponds to the input dim of the LSTM in Set2Set. 
            This is typically the sum of the input dim and the lstm output dim. If not provided, it will be set to :obj:`input_dim*2`
        steps: int, optional
            Number of iterations :math:`T`. If not provided, the number of nodes will be used.
        num_layers : int, optional
            Number of recurrent layers (e.g., :obj:`num_layers=2` would mean stacking two LSTMs together)
            (Default, value = 1)
        activation: str, optional
            Activation function to apply after the pooling layer. No activation is used by default.
            (Default value = None)
    """

    def __init__(self, input_dim, hidden_dim=None, steps=None, num_layers=1, activation=None):
        super(Set2Set, self).__init__()
        self.steps = steps
        self.input_dim = input_dim
        self.hidden_dim = input_dim * 2 if hidden_dim is None else hidden_dim
        if self.hidden_dim <= self.input_dim:
            raise ValueError(
                'Set2Set hidden_dim should be larger than input_dim')
        # the hidden is a concatenation of weighted sum of embedding and LSTM output
        self.lstm_output_dim = self.hidden_dim - self.input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_dim, self.input_dim,
                            num_layers=num_layers, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self._activation = None
        if activation:
            self._activation = get_activation(activation)

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor 
                Input tensor of size (B, N, D)

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the  set2set pooling operation.
        """

        batch_size = x.shape[0]
        n = self.steps or x.shape[1]

        h = (x.new_zeros((self.num_layers, batch_size, self.lstm_output_dim)),
             x.new_zeros((self.num_layers, batch_size, self.lstm_output_dim)))

        q_star = x.new_zeros(batch_size, 1, self.hidden)

        for i in range(n):
            # q: batch_size x 1 x input_dim
            q, h = self.lstm(q_star, h)
            # e: batch_size x n x 1
            e = torch.matmul(x, torch.transpose(q, 1, 2))
            a = self.softmax(e)
            r = torch.sum(a * x, dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=-1)

        if self._activation:
            return self._activation(q_star, dim=1)
        return torch.squeeze(q_star, dim=1)