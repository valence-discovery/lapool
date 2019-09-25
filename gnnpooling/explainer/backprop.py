import torch
import types
import numpy as np
from torch.autograd import Variable, Function

class VanillaGradExplainer(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def _backprop(self, inp, ind, grad_ind):
        inp = [x.unsqueeze(0).requires_grad_() for x in inp]
        with torch.autograd.detect_anomaly():
            output = torch.sigmoid(self.model(inp))
            # Zero grads
            self.model.zero_grad()
            if ind is None:
                ind = [int(output.argmax(dim=-1)[0])]
            if not isinstance(ind, list):
                ind = [ind]
            ind = torch.tensor(ind).long()
            # Target for backprop
            grad_out = torch.zeros_like(output)
            grad_out.scatter_(-1, ind.unsqueeze(0).t(), 1)
            # Backward pass
            output.backward(gradient=grad_out, retain_graph=True)
            return inp[grad_ind].grad

    def explain(self, inp, ind=None, grad_ind=-1):
        return self._backprop(inp, ind, grad_ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GradxInputExplainer, self).__init__(model)

    def explain(self, inp, ind=None, grad_ind=-1):
        grad = self._backprop(inp, ind, grad_ind)
        return inp[grad_ind] * grad


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(SaliencyExplainer, self).__init__(model)

    def explain(self, inp, ind=None, grad_ind=-1):
        grad = self._backprop(inp, ind, grad_ind)
        return grad.abs()


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100):
        super(IntegrateGradExplainer, self).__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None, grad_ind=-1):
        grad = 0
        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = [x.clone().requires_grad_(False) for x in inp]
            new_inp[grad_ind] = new_inp[grad_ind]*alpha
            g = self._backprop(new_inp, ind, grad_ind)
            grad += g
        return grad * inp[grad_ind] / self.steps


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(DeconvExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


class GuidedBackpropExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GuidedBackpropExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float()
                mask2 = (grad_output > 0).float()
                grad_inp = mask1 * mask2 * grad_output
                grad_output.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


# modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
class SmoothGradExplainer(object):
    def __init__(self, model, base_explainer, stdev_spread=0.15,
                nsamples=25, magnitude=True):
        self.model = model
        self.base_explainer = base_explainer(self.model)
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude

    def explain(self, inp, ind=None, grad_ind=-1):
        stdev = self.stdev_spread * (inp[grad_ind].max() - inp[grad_ind].min())
        total_gradients = 0
        origin_inp_data = inp[grad_ind].clone()

        for i in range(self.nsamples):
            noise = torch.randn(inp[grad_ind].size()) * stdev
            inp[grad_ind].copy_(noise + origin_inp_data)
            grad = self.base_explainer.explain(inp, ind, grad_ind)
            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples


class GradCAMExplainer(VanillaGradExplainer):
    def __init__(self, model, target_layer, use_inp=False):
        super(GradCAMExplainer, self).__init__(model)
        self.target_layer = target_layer
        self.use_inp = use_inp
        self.intermediate_act = []
        self.intermediate_grad = []
        self._register_forward_backward_hook()

    def _register_forward_backward_hook(self):
        def forward_hook_input(m, i, o):
            self.intermediate_act.append(i[0].clone())

        def forward_hook_output(m, i, o):
            self.intermediate_act.append(o.clone())

        def backward_hook(m, grad_i, grad_o):
            self.intermediate_grad.append(grad_o[0].clone())

        if self.use_inp:
            self.target_layer.register_forward_hook(forward_hook_input)
        else:
            self.target_layer.register_forward_hook(forward_hook_output)

        self.target_layer.register_backward_hook(backward_hook)

    def _reset_intermediate_lists(self):
        self.intermediate_act = []
        self.intermediate_grad = []

    def explain(self, inp, ind=None, grad_ind=-1):
        self._reset_intermediate_lists()

        _ = super(GradCAMExplainer, self)._backprop(inp, ind, grad_ind)

        grad = self.intermediate_grad[0]
        act = self.intermediate_act[0]

        weights = grad.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        cam = weights * act
        cam = cam.sum(1).unsqueeze(1)

        cam = torch.clamp(cam, min=0)

        return cam

