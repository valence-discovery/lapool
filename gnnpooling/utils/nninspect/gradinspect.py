import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from poutyne.framework.callbacks import Callback
from collections import defaultdict, OrderedDict
import logging

logger = logging.getLogger("GradInspect")


class GradientInspector(Callback):
    r"""
    Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        and to visualize the gradient flow. 

        Arguments
        ---------
            bottom_zoom: float, default=-0.001
                Min value of the gradient to be shown
            top_zoom: float, default=0.02
                Max value of the gradient to be shown. Use this conjointly with the bottom_zoom to zoom into
                only part of the plot. Please keep in mind that only the abs value of the gradient matter.
            max_norm: float, default=4
                Max range of the gradient norm to display
            speed: float, default=0.05
                Time to wait before updating the plot
            log: bool, default=False
                Whether to print in terminal
            update_at: str, default="backward"
                When should the plot be updated ? One of {backward, epoch, batch}. 
            figsize: tuple, default=(10, 6)
                Size of the plot
            showbias: bool, default=False
                Whether bias should be considered
    """

    def __init__(self, bottom_zoom=-0.001, top_zoom=0.02, max_norm=4, speed=0.05, log=False, update_at="backward", figsize=(10, 6), showbias=False):
        super().__init__()
        self.fig, self.axes = plt.subplots(ncols=2, sharey=True, tight_layout=True, gridspec_kw={
                                           'width_ratios': [3, 1]}, figsize=figsize)
        self.speed = speed
        self.top_zoom = top_zoom
        self.bottom_zoom = bottom_zoom
        self.updater = update_at
        self.bias = showbias
        self.max_norm = max_norm
        self._configure_logger(log)

    def _configure_logger(self, log):
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        if log:
            ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def log_grad(self, params, header="Epoch", padding=0):
        logger.debug("\t"*padding + "==> "+header)
        for name, param in params.items():
            # if param.grad is not None:
            # logger.debug("\t"*padding + f"{name}: No grad")
            logger.debug("\t"*padding +
                         "{}: {}".format(name, param.abs().mean()))

    def on_epoch_begin(self, epoch, logs=None):
        self.parameters_means = defaultdict(list)

    def on_batch_end(self, batch, logs=None):
        grads = OrderedDict()
        for name, param in self.model.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.data.cpu().clone()
                self.parameters_means[name].append(grads[name])
        self.log_grad(grads, "Batch: {}".format(batch), padding=1)
        if self.updater == "batch":
            try:
                self.update(grads)
            except KeyboardInterrupt:
                raise
            except:
                pass

    def on_epoch_end(self, epoch, logs=None):
        grads = OrderedDict((name, torch.stack(params))
                            for name, params in self.parameters_means.items())
        self.log_grad(grads, "Epoch")
        if self.updater == "epoch":
            try:
                self.update(grads)
            except KeyboardInterrupt:
                raise
            except:
                pass

    def on_backward_end(self, batch):
        r"""Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        and to visualize the gradient flow"""
        if self.updater == "backward":
            grads = OrderedDict((name, param.grad.data.cpu(
            )) for name, param in self.model.model.named_parameters() if param.grad is not None)
            try:
                self.update(grads)
            except KeyboardInterrupt:
                raise
            except:
                pass

    def update(self, grads):
        ave_grads = []
        max_grads = []
        layers = []
        norm_grads = []

        for ax in self.axes.flat:
            ax.clear()

        for n, p in grads.items():
            if ("bias" not in n or self.bias):
                layers.append(n + " ("+",".join(str(x) for x in p.shape) + ")")
                ave_grads.append(p.abs().mean())
                max_grads.append(p.abs().max())
                norm_grads.append(p.norm())
        self.axes[0].barh(np.arange(len(max_grads)), max_grads,
                          height=0.6, zorder=10, alpha=0.4, lw=1, color="#A3D6D2")
        self.axes[0].barh(np.arange(len(max_grads)), ave_grads,
                          height=0.6, zorder=10, alpha=0.9, lw=1, color="#2F6790")
        self.axes[0].vlines(0, -0.5, len(ave_grads)+1, lw=2, color="k")
        self.axes[1].barh(np.arange(len(max_grads)), norm_grads,
                          height=0.6, zorder=10, alpha=0.5, color="gray")
        self.axes[0].invert_xaxis()
        self.axes[0].set(title='Gradient values')
        self.axes[1].set(title='Gradient norm')
        self.axes[0].set_ylim(bottom=-0.5, top=len(ave_grads))
        # zoom in on the lower gradient regions
        self.axes[0].set_xlim(right=self.bottom_zoom, left=self.top_zoom)
        self.axes[0].yaxis.set_tick_params(
            left=False, labelright=False, labelleft=False)
        self.axes[1].yaxis.set_tick_params(
            labelright=True, labelleft=False, left=False, right=True, labelsize=9)
        self.axes[1].set(yticks=range(0, len(ave_grads)), yticklabels=layers)
        # zoom in on the lower gradient regions
        self.axes[1].set_xlim(left=0, right=self.max_norm)

        for ax in self.axes.flat:
            ax.margins(0.01)
            ax.grid(True)

        self.fig.legend([Line2D([0], [0], color="#A3D6D2", lw=3),
                         Line2D([0], [0], color="#2F6790", lw=3), Line2D([0], [0], color="gray", lw=3)], ['max-gradient', 'mean-gradient', "norm"], loc='upper right', bbox_to_anchor=(1, 1), fontsize=9)
        self.fig.subplots_adjust(wspace=0.023)
        plt.pause(self.speed)
