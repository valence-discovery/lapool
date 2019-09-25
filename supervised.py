import click
import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
import deepchem as dc
import torch.nn.functional as F
import yaml

from functools import partial
from poutyne import torch_to_numpy, numpy_to_torch
from sklearn import metrics as skmetrics

from gnnpooling.models.layers import FCLayer, get_activation
from gnnpooling.pooler.batchlapool import LaPool
from gnnpooling.pooler.diffpool import DiffPool
from gnnpooling.pooler.topkpool import TopKPool

from gnnpooling.utils.data import NetworkXGraphDataset, read_dense_graph_data as read_graph, load_mol_dataset as load_dataset
from gnnpooling.models.layers import FCLayer, GlobalSoftAttention, GlobalSumPool1d, GlobalGatedPool, GlobalGatedPool, DeepSetEncoder, GlobalAvgPool1d
from gnnpooling.models.gcn import GINLayer, GCNLayer
from gnnpooling.utils.trainer import Trainer
from gnnpooling.utils.metrics import roc_auc_score, accuracy
from gnnpooling.utils.tox21data import get_tox21, as_dataset, load_chembl


def get_hpool(**params):
    arch = params.pop("arch", None)
    if arch is None:
        return None, None
    if arch == "diff":
        pool = partial(DiffPool, **params)
    elif arch == "topk":
        pool = partial(TopKPool, **params)
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
    return GlobalSumPool1d()


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
        return h, side_loss


class STrainer(Trainer):
    def _compute_loss_and_metrics(self, x, y, *, return_loss_tensor=False, return_pred=False):
        (adj, x, mask), *y = self._process_input(x, *y)
        mols, y, *w = y
        if len(w):
            w = w[0]
        else:
            w = None
        pred_y, side_loss = self.model(x, adj, mask)
        loss = self.loss_function(pred_y, y, weights=w) + side_loss
        if not return_loss_tensor:
            loss = float(loss)

        with torch.no_grad():
            metrics = self._compute_metrics(pred_y, y)
        pred_y = torch_to_numpy(pred_y) if return_pred else None
        return loss, metrics, pred_y

    def _compute_metrics(self, pred_y, y):
        if isinstance(pred_y, tuple):
            pred_y = pred_y[0]
        return np.array([float(metric(pred_y.detach(), y)) for metric in self.metrics])


def loss_fn(pred, targets, weights=None, base_criterion=None):
    if weights is not None:
        base_criterion.reduction = 'none'
        loss = base_criterion(pred, targets) * weights.detach()
        return loss.mean()
    return base_criterion(pred, targets)



def get_dataset(dataset, min_size=0, max_size=None, **kwargs):
    if dataset == 'tox21':
        return get_tox21(min_size=min_size, max_size=max_size, **kwargs)

    elif dataset == 'fragments':
        data_path = 'chembl_dataset_fragments.txt'
        t, (train, valid, test), _ = load_chembl(dataset_file=data_path)
        return as_dataset(train, valid, test, min_size=min_size, max_size=max_size, **kwargs)


    elif dataset == 'alerts':
        data_path = 'chembl_dataset_alerts.txt'
        t, (train, valid, test), _ = load_chembl(dataset_file=data_path)
        return as_dataset(train, valid, test, min_size=min_size, max_size=max_size, **kwargs)

    else:
        x, y = read_graph(dataset, min_num_nodes=min_size, max_num_nodes=max_size)
        train_dt, valid_dt, test_dt = load_dataset(X, y, test_size=test_size, device=device,
                                                   valid_size=valid_size,
                                                   shuffle=True, balance=balance,
                                                   with_bond=with_bond, mol_filter=mol_filter, type=graph_type,
                                                   stratify=stratify)
        raise ValueError



def get_loss(dataset):    
    return partial(loss_fn, base_criterion=nn.BCEWithLogitsLoss(reduction='mean'))


def save_model(model, output):
    joblib.dump(model, output)

@click.command()
@click.option('--arch', '-a', default='gnn', help="Type of model")
@click.option('--dataset', '-d', default='tox21', help="Supervised dataset")
@click.option('--max_nodes', default=50, type=int, help="Maximum number of nodes")
@click.option('--min_nodes', default=5, type=int, help="Minimum number of nodes")
@click.option('--ksize', '-k', default=0.125, type=float, help="Percentage of nodes to retains during hierarchical pooling")
@click.option('--config_file', '-c', required=True, type=click.Path(exists=True), help="File containing the configuration file")
@click.option('--hparams', '-h', type=click.Path(exists=True), help="File containing the hpool params")
@click.option('--output_path', '-o', default="", help="Output folder")
def cli(arch, dataset, max_nodes, min_nodes, ksize, config_file, hparams, output_path):
    hpool_params = {}
    hpool_params_add = {}
    hidden_dim = int(ksize * max_nodes)
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
    METRICS = {'acc': accuracy}
    arch = f"{arch}_{hidden_dim}"

    for repeat in range(3):
        np.random.seed(repeat)
        torch.manual_seed(repeat)
        model_dir = os.path.join(output_path, arch, str(repeat))
        os.makedirs(model_dir, exist_ok=True)
        train_dt, valid_dt, test_dt, generator, in_size, out_size = get_dataset(dataset, min_size=min_nodes, max_size=max_nodes)
        loss = get_loss(dataset)
        model = Net(max_nodes, in_size, out_size, config=config)
        print(in_size, out_size)
        trainer = STrainer(model, loss_fn=loss, metrics=METRICS,
                           gpu=True, model_dir=model_dir, op__lr=5e-4)
        trainer.fit(train_dt, valid_dt, batch_size=64, epochs=100, generator_fn=generator, checkpoint='model_{}.pth.tar'.format(arch), tboardX="logs",
                    checkpoint__restore_best=True, reduce_lr={"verbose": True, "factor": 0.5, "cooldown": 3}, early_stopping={"patience": 20, "min_delta": 1e-3})
        print("Evaluation step")
        loss, metric, pred = trainer.evaluate(
            test_dt, batch_size=64, return_pred=True, generator_fn=generator)
        y_pred = torch.sigmoid(torch.Tensor(pred)).numpy()
        print("Test results")
        metric_results = dict()
        for metric_name, metric in {'acc': accuracy, 'roc': partial(roc_auc_score, per_tasks=True)}.items():
            try:
                m_value = metric(y_pred, test_dt.y)
                metric_results[metric_name] = m_value
                if metric_name in ['roc']:
                    metric_results[metric_name + "_mean"] = np.mean(m_value)
            except:
                pass

        np.savez_compressed(os.path.join(
            model_dir, 'test_{}.npz'.format(arch)), test=test_dt.y, pred=y_pred)
        joblib.dump(metric_results, os.path.join(
            model_dir, "metric_{}".format(arch)))
        save_model(trainer.model, os.path.join(
            model_dir, "models_{}.pkl".format(arch)))
        print(metric_results)
        print("=================")


if __name__ == '__main__':
    cli()
