import json
import os
import torch
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')
from gnnpooling.runner import fingerprint_multitask_loss, save_model
from functools import partial
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from gnnpooling.utils.metrics import roc_auc_score, accuracy, f1_score
from gnnpooling.utils import const
from gnnpooling.utils.trainer import Trainer
from gnnpooling.utils.data import *
from gnnpooling.configs import gen_model_getter, sup_model_getter
from gnnpoolingmodelsnetworks import Discriminator, Encoder, Decoder, MLPdiscriminator
from gnnpoolingmodelsgan import GAN
from gnnpoolingmodelsaae import AAE
from gnnpooling.scoring import MolMetrics
import torch.nn as nn
import torch.nn.functional as F
from ivbase.utils.datasets.datacache import DataCache

import tarfile
import pandas as pd
import re

import pickle

EVAL_METRICS = {'acc': accuracy, 'f1_macro_sk': partial(f1_score, average='macro'),
                'roc': roc_auc_score, 'f1_micro': partial(f1_score, average='micro'),
                'f1_macro': partial(f1_score, average='macro'),
                'f1_weighted': partial(f1_score, average='weighted')}


def get_best_model_path(path):
    outs = os.listdir(path)
    epochs = []
    for i, name in enumerate(outs):
        r = re.search('\d\d', name)
        if r is not None:
            epochs.append((i, float(r.group(0))))
    best_epoch = epochs[-1]
    model_path = os.path.join(path, outs[best_epoch[0]])

    return model_path


def get_model_params(model, name, tar=None):
    if tar is not None:
        with tar.extractfile(model) as f:
            model = pickle.load(f)
    else:
        with open(model, 'rb') as f:
            model = pickle.load(f)
    keys = filter(lambda x: not x.startswith('_') and x not in ['training', 'pool_inspect', 'pooling_loss'],
                  model.__dict__.keys())
    params = {x: model.__dict__[x] for x in keys}
    pool_arch = recover_pool_arch(name)
    return params


def get_model_paths(results_paths, results_dir):
    model_paths = []
    parameters = []
    for key in results_paths:
        for path in results_paths[key]:
            try:
                path = os.path.join(results_dir, path, 'output')
                if os.path.exists(os.path.join(path, 'model')):
                    model_path = get_best_model_path(os.path.join(path, 'model'))
                    outs = os.listdir(os.path.join(path, 'model'))
                    for name in outs:
                        if name.endswith('.pkl') and name.startswith('model'):
                            params = get_model_params(os.path.join(path, 'model', name), name)
                else:
                    tar = tarfile.open(path + '/model.tar')
                    outs = tar.getnames()
                    for name in outs:
                        if name.endswith('.pkl') and name.startswith('model'):
                            params = get_model_params(name, name, tar)
                    epochs = []
                    for i, name in enumerate(outs):
                        r = re.search('\d\d', name)
                        if r is not None:
                            epochs.append((i, float(r.group(0))))
                    epochs.sort(key=lambda x: x[1])
                    best_epoch = epochs[-1]
                    model_path = outs[best_epoch[0]]
                    model = tar.getmember(model_path)
                    tar.extract(model, path=path + '/model')
                    print(os.listdir(path + '/model'))
                    model_path = os.path.join(path, model, os.listdir(path + '/model')[0])
                model_paths.append(model_path)
                parameters.append(params.copy())
            except FileNotFoundError:
                pass
    assert len(model_paths) == len(parameters)
    return model_paths, parameters


def recover_pool_arch(name):
    name = re.search(":*\.", name).group(0).strip(':.')
    configs = sup_model_getter(1)
    pool_params = configs[name]['pool_arch']
    return pool_params, name


def compute_metrics(clf, y_pred, targets, output_path, model_name):
    eval_metrics = EVAL_METRICS.keys()
    metrics_results = defaultdict(list)
    results = []
    for metric_name in eval_metrics:
        metric = EVAL_METRICS[metric_name]

        # IF ROC, recover from error using macro average over tasks, with 0.5 task score as default when undefined.
        if metric_name == 'roc':
            try:
                m_value = metric(y_pred, targets)
            except ValueError as e:
                if len(targets.shape) == 1:
                    m_value = 0.5
                elif len(targets.shape) == 2:
                    scores = []
                    for col in range(targets.shape[1]):
                        if len(np.unique(targets[:, col])) == 1:
                            scores.append(0.5)
                        else:
                            scores.append(metric(y_pred[:, col], targets[:, col]))
                    m_value = np.asarray(scores).mean()
                else:
                    raise e
        elif metric_name in ['f1_macro', 'f1_weighted']:
            scores = []
            for col in range(targets.shape[1]):
                scores.append(metric(y_pred[:, col], targets[:, col]))
            m_value = np.asarray(scores).mean()
        else:
            m_value = metric(y_pred, targets)
        metrics_results[metric_name].append(m_value)

    """
    np.savez_compressed(os.path.join(output_path, 'test_{}_{}.npz'.format(model_name)),
                        test=targets,
                        pred=y_pred)
    save_model(clf, os.path.join(output_path, "model:{}_{}.pkl".format(model_name)))
    """
    tmp_results = dict([('name', model_name)])
    for name, vals in metrics_results.items():
        tmp_results[name] = vals
    print(tmp_results)
    results.append(tmp_results)
    """" "
    with open(os.path.join(output_path, "data_{}.pkl".format(model_name)), 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)
    """


def main(config, output_path, results_paths, results_dir):
    seed = int(config.pop('seed', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """Run inductive learning models"""
    METRICS = {'acc': accuracy}

    # Retrieve YAML file id
    config_id = int(config.pop("config", 1))
    with_bond = bool(config.get("with_bond", False))
    max_n = int(config.get("max_n", -1))
    num_epochs = int(config.get("epochs", 10))
    model_params = sup_model_getter(config_id)
    input_path = str(config.get("data_path", "standard_data/tox21/tox21.csv"))
    mol_filter = int(config.get("filter_size", 0))
    balance = bool(config.get('balance', False))
    stratify = bool(config.get('stratify', False))

    n_out = int(config.pop("tasks", -1))
    repeats = int(model_params.pop('repeats', 1))
    file_type = model_params.pop('filetype', 'csv')
    test_size = model_params.pop('test_size', 0.2)
    valid_size = model_params.pop('test_size', 0.2)
    shuffle = bool(model_params.pop('shuffle', True))
    device = model_params.pop('device', ('cuda' if torch.cuda.is_available() else 'cpu'))
    model_name = str(config.get("name", 'gcn'))
    mparams = model_params[model_name]
    if not mol_filter or model_name != 'gcn_laplacian':
        mol_filter = mparams.get('pool_arch', {}).get('hidden_dim', 0)
    if model_name == 'gcn_laplacian':
        reg_mode = int(config.get("reg_mode", 1))
        hop = int(config.get("hop", 3))
        lap_hop = int(config.get("lap_hop", 1))
        init_params = dict(reg_mode=reg_mode, hop=hop, lap_hop=lap_hop)
        mparams['pool_arch'].update(init_params)

    # Set up data
    import time
    start = time.time()
    X, y, *meta = read_data(input_path, n_out, format=file_type, max_n=max_n)
    train_dt, valid_dt, test_dt = load_mol_dataset(X, y, test_size=test_size, device=device,
                                                         valid_size=valid_size,
                                                         shuffle=shuffle, balance=balance,
                                                         with_bond=with_bond, mol_filter=mol_filter, stratify=stratify)
    in_size = train_dt[0][0][1].shape[-1]
    nedges = train_dt[0][0][0].shape[-1]  # can be either 1 or n_edges
    out_dim = train_dt[0][1].shape[-1]
    end = time.time()
    print("Data processing took {:.2f} seconds".format(end - start))

    results = []
    model = Encoder
    batch_size = 32
    test_batch_size = 32
    step_train = int(np.ceil(len(train_dt) / batch_size))
    step_valid = int(np.ceil(len(valid_dt) / batch_size))
    step_test = int(np.ceil(len(test_dt) / batch_size))
    train_generator = partial(batch_generator_sup, balance=balance)
    test_generator = partial(batch_generator_sup, infinite=False, shuffle=False, balance=balance)

    model_paths, parameters = get_model_paths(results_paths=results_paths, results_dir=results_dir)

    for model_path, params in zip(model_paths, parameters):
        print(model_path)
        state_dict = torch.load(model_path)['net']
        pool_arch, model_name = recover_pool_arch(state_dict, params['pool_arch'])
        params.update(dict(input_dim=in_size, out_dim=out_dim, nedges=nedges, pool_arch=pool_arch))
        clf = model(**params)

        trainer = Trainer(clf, loss_fn=fingerprint_multitask_loss, metrics=METRICS, gpu=(device == "cuda"),
                          model_dir=output_path)
        clf.load_state_dict(state_dict)
        clf.eval()
        res = trainer.evaluate(test_dt, batch_size=test_batch_size, return_pred=True, generator_fn=test_generator,
                               steps=step_test)

        y_pred = F.sigmoid(torch.Tensor(res[2])).numpy()

        compute_metrics(clf, y_pred, test_dt.y, output_path, model_name)


if __name__ == '__main__':

    results_cache = DataCache(cache_root='/Users/julienhorwood/Documents/invivo/supervised_final_caches')
    results_dir = results_cache.get_dir("s3://invivoai-sagemaker-artifacts/molg")

    algo_names = ['gcn-alert',
                  'diffpool-alert',
                  'topk-alert',
                  'laplacian-alert']
    results_paths = {}
    for algo in algo_names:
        for d in os.listdir(results_dir):
            if d.startswith(algo):
                if algo in results_paths:
                    results_paths[algo].append(d)
                else:
                    results_paths[algo] = [d]

    config = {
        "supervised": 1,
        "name": "gcn",
        "epochs": 10,
        "with_bond": True,
        "max_n": -1,
        "data_path": "../data/dataset_alert.txt",
        "tasks": 55,
        "filter_size": 3,
        "balance": False
    }
    main(config, 'test_results', results_paths, results_dir)
