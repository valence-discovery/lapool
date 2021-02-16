import click
import os
import json
import numpy as np
import sys
import torch
import torch.nn as nn
import joblib
import torch.nn.functional as F
import yaml
import traceback

from collections import defaultdict
from functools import partial
from sklearn import metrics as skmetrics

from gnnpooling.models.gcn import GINLayer, GCNLayer
from gnnpooling.models.layers import FCLayer, get_activation
from gnnpooling.pooler import get_gpool, get_hpool
from gnnpooling.utils.trainer import SupervisedTrainer
from gnnpooling.utils.metrics import roc_auc_score, accuracy, f1_score, get_loss
from gnnpooling.utils.read_data import load_supervised_dataset
from gnnpooling.utils.datasets import GraphDataLoader


class GNNBlock(nn.Module):
    def __init__(self, g_size, in_size, hidden_size=[128, 128], residual=False, b_norm=True, dropout=0., activation="relu", bias=False):
        super(GNNBlock, self).__init__()
        self.G_size = g_size
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.conv_layers = nn.ModuleList()
        self.b_norm = b_norm
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.res_net = None
        for ksize in self.hidden_size:
            gc = GINLayer(in_size, ksize, G_size=self.G_size, b_norm=self.b_norm,
                          dropout=self.dropout, activation=self.activation, bias=bias)
            self.conv_layers.append(gc)
            in_size = ksize
        self.out_size = in_size + self.residual * self.in_size

    def forward(self, G, x, mask):
        h = x
        for cv_layer in self.conv_layers:
            G, h = cv_layer(G, h, mask=mask)
        if self.residual:
            h = torch.cat([h, x], dim=-1)
        return G, h


class Net(nn.Module):
    def __init__(self, g_size, in_size, out_size, config):
        super(Net, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.conv1 = GNNBlock(g_size, in_size, **config["conv_before1"])
        self.conv2 = GNNBlock(g_size, self.conv1.out_size,
                              **config["conv_before2"])
        self.hpool, self.hpool_arch = get_hpool(**config["hpool"])
        hsize = self.conv2.out_size
        if self.hpool_arch is not None:
            self.hpool = self.hpool(input_dim=hsize)
        if self.hpool_arch == 'laplacian':
            hsize = self.hpool.cluster_dim

        self.conv3 = GNNBlock(None, hsize, **config["conv_after1"])

        in_size = self.conv3.out_size
        self.gpooler = get_gpool(input_dim=in_size, **config["gpool"])

        self.fc_layers = nn.ModuleList()
        for conf in config["fclayers"]:
            fc = FCLayer(in_size=in_size, **conf)
            self.fc_layers.append(fc)
            in_size = conf["out_size"]
        self.out_layers = nn.Linear(in_size, self.out_size)

    def forward(self, x, G, mask):
        G, x = self.conv1(G, x, mask)
        G, h = self.conv2(G, x, mask)

        side_loss = 0
        if not self.hpool_arch:
            G, hpool = G, h
        elif self.hpool_arch == 'diff':
            G, hpool, side_loss = self.hpool(
                G, z=h, x=x, return_loss=True, mask=mask)
        else:
            G, hpool, side_loss = self.hpool(G, h, return_loss=True, mask=mask)

        G, h = self.conv3(G, hpool, mask=None)
        h = self.gpooler(h)
        for fc_layer in self.fc_layers:
            h = fc_layer(h)
        h = self.out_layers(h)
        # side_loss is optional
        return h, side_loss


@click.command()
@click.option('--arch', '-a', default='laplacian', help="Type of model")
@click.option('--dataset', '-d', default='tox21', help="Supervised dataset")
@click.option('--max_nodes', default=50, type=int, help="Maximum number of nodes")
@click.option('--min_nodes', default=5, type=int, help="Minimum number of nodes")
@click.option('--ksize', '-k', default=0.10, type=float, help="Percentage of nodes to retains during hierarchical pooling")
@click.option('--config_file', '-c', required=True, type=click.Path(exists=True), help="File containing the model global configuration file")
@click.option('--hparams', '-h', type=click.Path(exists=True), help="File containing the hpool params")
@click.option('--output_path', '-o', default="", help="Output folder")
@click.option('--repeats', '-r', type=int, default=1, help="Number of repeat to perform under alternative splits")
@click.option('--epochs', '-e', type=int, default=100, help="Number of epochs")
@click.option('--batch_size', '-b', type=int, default=32, help="Batch size")
@click.option('--cpu', is_flag=True, help="Force use cpu")
@click.option('--no_early_stopping', is_flag=True, help="Don't use early stoppping")
def cli(arch, dataset, max_nodes, min_nodes, ksize, config_file, hparams, output_path, repeats, epochs, batch_size, cpu, no_early_stopping):
    torch.backends.cudnn.benchmark = True
    hpool_params = {}
    hpool_params_add = {}
    hidden_dim = int(ksize * max_nodes) or None
    if hparams:
        with open(hparams) as IN:
            hpool_params_add.update(json.load(IN))
        dataset = hpool_params_add.pop("task", dataset)
        arch = hpool_params_add.get("arch", arch)
    if 'topk' in arch:
        hpool_params = dict(arch="topk", hidden_dim=hidden_dim)
    elif 'diff' in arch:
        hpool_params = dict(arch="diff", hidden_dim=hidden_dim)
    elif 'lap' in arch:
        hpool_params = dict(
            arch="laplacian", hidden_dim=hidden_dim, cluster_dim=128)
    with open(config_file) as IN:
        config = yaml.safe_load(IN)

    hpool_params.update(hpool_params_add)
    hidden_dim = hpool_params.get("hidden_dim", hidden_dim)
    config["hpool"].update(hpool_params)
    METRICS = {'acc': accuracy} # classification accuracy as metric
    arch = f"{arch}_{hidden_dim}"
    early_stopping = {"patience": epochs//5, "min_delta": 1e-4}
    if no_early_stopping:
        early_stopping = None
    for repeat in range(repeats):
        np.random.seed(repeat)
        torch.manual_seed(repeat)
        model_dir = os.path.join(output_path, arch, str(repeat))
        os.makedirs(model_dir, exist_ok=True)
        (train_dt, valid_dt, test_dt), in_size, out_size = load_supervised_dataset(dataset, min_size=min_nodes, max_size=max_nodes)
        generator = partial(GraphDataLoader, drop_last=True)
        loss = get_loss(dataset)
        model = Net(max_nodes, in_size, out_size, config=config)
        
        SIGMOID = (dataset.upper() in ['ALERTS', 'FRAGMENTS', 'TOX21'])
        print(f"==> Training step ({repeat+1}/{repeats})")
        trainer = SupervisedTrainer(model, loss_fn=loss, metrics=METRICS,
                                    gpu=(not cpu), model_dir=model_dir, op__lr=1e-4, sigmoid=SIGMOID)
        trainer.fit(train_dt, valid_dt, batch_size=batch_size, epochs=epochs, generator_fn=generator, checkpoint='model_{}.pth.tar'.format(arch), tboardX="logs",
                    checkpoint__restore_best=True, reduce_lr={"verbose": True, "factor": 0.5, "cooldown": 3}, early_stopping=early_stopping)
        print(f"==> Evaluation step ({repeat+1}/{repeats})")
        loss, metric, pred = trainer.evaluate(
            test_dt, batch_size=len(test_dt)//10, return_pred=True, generator_fn=GraphDataLoader)
        loss_val, metric_val, pred_val = trainer.evaluate(
            valid_dt, batch_size=len(valid_dt)//10, return_pred=True, generator_fn=GraphDataLoader)
        if SIGMOID:
            y_pred = torch.sigmoid(torch.Tensor(pred)).numpy()
            y_pred_val = torch.sigmoid(torch.Tensor(pred_val)).numpy()
        else:
            y_pred = torch.softmax(torch.Tensor(pred), dim=-1).numpy()
            y_pred_val = torch.softmax(torch.Tensor(pred_val), dim=-1).numpy()
        metric_results = defaultdict(dict)
        EVAL_METRICS = {'acc': accuracy, 'roc': partial(roc_auc_score, per_tasks=True), 'roc_macro':partial(roc_auc_score, average="macro")}
        if dataset.upper() in ['ALERTS', 'FRAGMENTS']:
            EVAL_METRICS = {'acc': accuracy, 'roc_micro':partial(roc_auc_score, average="micro"), 'f1_micro':partial(f1_score, average="micro")}

        for (dt, dtname, pred) in zip([valid_dt, test_dt], ['valid', 'test'], [y_pred_val, y_pred]):
            for metric_name, metric in EVAL_METRICS.items():
                try:
                    m_value = metric(pred, dt.y)
                    metric_results[dtname][metric_name] = m_value
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
            np.savez_compressed(os.path.join(
                model_dir, '{}_{}.npz'.format(dtname, arch)), test=dt.y, pred=pred)
            
        joblib.dump(metric_results, os.path.join(model_dir, "metric_{}".format(arch)))
        # dump model
        joblib.dump(trainer.model, os.path.join(model_dir, "models_{}.pkl".format(arch)))
        print(metric_results)
        print(f"================= {repeat+1}")


if __name__ == '__main__':
    cli()
