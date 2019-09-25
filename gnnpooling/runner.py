import json
import os
import torch
import numpy as np
import warnings
import sklearn.metrics as skmetrics

warnings.filterwarnings(action='ignore')

from functools import partial
from collections import defaultdict
from gnnpooling.utils.scalers import get_scaler
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from gnnpooling.utils.nninspect.gradinspect import GradientInspector
from gnnpooling.utils.metrics import _convert_to_numpy
from gnnpooling.utils import const
from gnnpooling.utils.trainer import Trainer
from gnnpooling.utils.data import *
from gnnpooling.configs import gen_model_getter, sup_model_getter
from gnnpooling.utils.trainer import SupervisedTrainer
from gnnpooling.models.networks import Discriminator, Encoder, Decoder, MLPdiscriminator
from gnnpooling.models.gan import GAN
from gnnpooling.models.aae import AAE
from gnnpooling.scoring import MolMetrics
import torch.nn as nn
import torch.nn.functional as F

import pickle


def save_model(model, output):
    try:
        model.save(output)
    except:
        joblib.dump(model, output)


def fingerprint_multitask_loss(pred, targets, weights=None, base_criterion=None, model=None, side_loss=False, binary=True):
    """
    Computes the Multi-task loss on the bitwise binary predictions.

    Arguments
    -------------

    pred: torch.Tensor
        prediction tensor of shape B x F (where F is the fingerprint length.)
    targets: torch.Tensor
        target tensor of shape B x F
    base_criterion: torch.nn.criterion
            Loss to be computed for each task prediction (bit) on the fingerprints.

    return: float
        aggregated multitask loss over bits

    """
    assert pred.dim() == 2
    if not binary:
        targets = targets.squeeze().long()
    if weights is not None:
        base_criterion.reduction = 'none'
        loss = base_criterion(pred, targets) * weights.detach()
        loss = loss.sum()
    else:
        loss = base_criterion(pred, targets)

    if side_loss:
        if model is None:
            side_loss = 0
        else:
            side_loss = model._sideloss()
        return loss / pred.shape[0] + side_loss
    return loss / pred.shape[0]


def run_experiment(model_params, output_path, input_path=None, **kwargs):
    """Run experiments"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    supervised = int(model_params.pop("supervised", 0))
    if supervised:
        return run_supervised_task(model_params, output_path, input_path, **kwargs)
    else:
        return run_gen_task(model_params, output_path, input_path, **kwargs)


def run_gen_task(config, output_path, input_path, **kwargs):
    """Run generative tasks"""

    # in generative mode
    const.ATOM_LIST = ['C', 'N', 'O', 'F']
    obj = int(config.get("tasks", 2))
    all_feat = int(config.get("all_feat", 0))
    with_bond = int(config.get("with_bond", 0))
    min_size = int(config.get("min_size", 0))
    max_size = int(config.get("max_size", 50))
    inspect = bool(int(config.get("inspect", 0)))
    l_gp = config.get("grad_pen", 0)
    smiles, y = read_data(input_path, obj)

    X, y, mols = transform_data(smiles, y, min_size=min_size, max_size=max_size, all_feat=all_feat, add_bond=with_bond)

    train_dt, valid_dt, test_dt = load_gen_dataset(X, y, mols, valid_size=0, cuda=const.CUDA_OK, pad_to=max_size)
    in_size = X[0][1].shape[-1]  # The size of the features
    nedges = X[0][0].shape[-1]  # can be either 1 or n_edges
    latent_dim = config.get("latent_dim", 128)
    n_mols = config.get("n_mols", 1000)

    atom_dim = GraphTransformer.atom_dim()
    feat_size = X[0][2].shape[-1]
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch", 32))
    fraction_per_epoch = 1
    step_train = int(len(train_dt) * fraction_per_epoch / batch_size)  # Number of training steps per epoch
    model_name = config.get("model", "GAN")
    generator = partial(batch_generator, split=feat_size)
    model_params = gen_model_getter(model_name, int(config.get("id", 1)))

    metrics = model_params.get("metrics", {"qed": MolMetrics.qed, "sas": MolMetrics.sas})
    metrics_norm = {"qed": partial(MolMetrics.qed, norm=True), "sas": partial(MolMetrics.sas, norm=True),
                    "noatom": MolMetrics.wrong_atoms, "logp": partial(MolMetrics.partition_coefficient, norm=True)}
    with_critic = int(model_params.get("with_critic", 0))

    grad_callback = []
    if inspect:
        grad_callback.append(GradientInspector(top_zoom=0.2, update_at="batch"))

    if model_name == "GAN":
        raise ValueError("Not supported anymore")
        # network = GAN(l_gp, sampler="soft_gumbel", metrics=metrics_norm)
        # discriminator = Discriminator(in_size, 1, nedges=nedges, feat_dim=feat_size, **model_params["discriminator"])
        # discriminator.set_inspector(inspect, show=False, outdir=os.path.join(output_path, model_name, "pooling"))
        # generator = Decoder(latent_dim, in_size, nedges=nedges, max_vertex=max_size, other_feat_dim=feat_size, **model_params["generator"])
        # network.set_generator(generator)
        # network.set_discriminator(discriminator)
        # if with_critic:
        #     critic = Discriminator(in_size, 1, nedges=nedges, feat_dim=feat_size,
        #                            **model_params.get("critic", model_params["discriminator"]))
        #     network.set_critic(critic)
        # trainer = GANTrainer(network, loss_fn=None, metrics=metrics, n_critic=5, D_op__lr=1e-2, G_op__lr=1e-2, gpu=const.CUDA_OK, model_dir=os.path.join(output_path, model_name)) # Define the trainer

    else:
        network = AAE(sampler="soft_gumbel", metrics=metrics_norm)
        encoder = Encoder(in_size, latent_dim, nedges=nedges, feat_dim=feat_size, **model_params["encoder"])
        encoder.set_inspector(inspect, show=True, outdir=os.path.join(output_path, model_name, "pooling"))
        decoder = Decoder(latent_dim, in_size, nedges=nedges, max_vertex=max_size, other_feat_dim=feat_size,
                          **model_params["decoder"])
        discriminator = MLPdiscriminator(latent_dim, **model_params["discriminator"])
        network.set_encoder(encoder)
        network.set_decoder(decoder)
        network.set_discriminator(discriminator)
        if with_critic:
            critic = MLPdiscriminator(latent_dim, **model_params.get("critic", model_params["discriminator"]))
            network.set_critic(critic)
        trainer = AAETrainer(network, loss_fn=None, metrics=metrics, gpu=const.CUDA_OK, D_op__lr=1e-2, G_op__lr=1e-2,
                             model_dir=os.path.join(output_path, model_name))
        trainer.set_training_mode(pretrain=True)
        trainer.fit(train_dt, None, epochs=epochs, steps_per_epoch=step_train, tboardX="logs",
                    validation_steps=None, generator_fn=generator, checkpoint=True, batch_size=batch_size, shuffle=True,
                    reduce_lr=False, callbacks=grad_callback, check__monitor="loss")
        trainer.set_training_mode(pretrain=False)

    grad_callback.append(
        GANScheduler(torch_lr_scheduler=torch.optim.lr_scheduler.StepLR, optim="dis", step_size=100, gamma=0.5))
    trainer.fit(train_dt, None, epochs=epochs, steps_per_epoch=step_train, tboardX="logs",
                validation_steps=None, generator_fn=generator, checkpoint=True, batch_size=batch_size, shuffle=True,
                reduce_lr=False, callbacks=grad_callback, check__monitor="loss")

    mol_generated = trainer.sample_molecules(n_gen)
    with open(os.path.join(outdir, "mols.txt"), "w") as MOL_OUT:
        MOL_OUT.write("\n".join(mol_generated))
    trainer.predict_generator((test_dt.mols, test_dt.X))  # Run the network on the testing set
    return y_pred

def accuracy(y_pred, y_true, weights=None):
    #print(y_pred)
    #print(y_true)
    if weights is not None:
        weights = weights.flatten()
    return skmetrics.accuracy_score(y_true.flatten(), y_pred.flatten(), sample_weight=weights)

def f1_score(y_pred, y_true, average='weighted'):
    return skmetrics.f1_score(y_true, y_pred, average=average)

def roc_auc_score(y_pred, y_true, average="macro", **kwargs):
    try:
        auc = skmetrics.roc_auc_score(y_true, y_pred, average=average)
    except Exception as e:
        print(e)
        auc = 0.5
    return auc


def score_fn(y_pred, y_true, *args, scorer=None, binary=False, auc=False, **kwargs):
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
    if not auc:
        if binary:
            y_pred = (np.asarray(y_pred) > 0.5).astype(int)
        else: 
            y_pred = np.argmax(y_pred, axis=1).astype(int)
    elif not binary:
        y_true = label_binarize(y_true, range(y_pred.shape[-1]))
    return scorer(y_pred, y_true, **kwargs)


def run_supervised_task(config, output_path, input_path, **kwargs):
    """Run inductive learning models"""
    # Reproducibility
    seed = int(config.pop('seed', 0))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Retrieve YAML file id
    config_id = int(config.pop("config", 1))
    with_bond = bool(config.get("with_bond", False))
    max_n = int(config.get("max_n", -1))
    num_epochs = int(config.get("epochs", 10))
    model_params = sup_model_getter(config_id)
    input_path = str(config.get("data_path", "data/tox21/tox21.csv"))
    mol_filter = int(config.get("filter_size", 0))
    balance = bool(config.get('balance', False))
    stratify = bool(config.get('stratify', False))
    repeats = int(config.pop('repeats', 1))
    valid_size = float(config.pop('valid_size', 0.15))
    max_num_nodes = config.pop('max_num_nodes', None)
    if max_num_nodes is not None:
        max_num_nodes = int(max_num_nodes)
    test_size = float(config.pop('test_size', 0.15))
    graph_type = config.pop('graph_type', 'dense')

    # Retrieve hyper-parameters
    n_out = int(config.pop("tasks", -1))
    file_type = model_params.pop('filetype', 'csv')

    shuffle = bool(model_params.pop('shuffle', True))
    device = model_params.pop('device', ('cuda' if torch.cuda.is_available() else 'cpu'))
    model_name = str(config.get("name", 'gcn'))
    mparams = model_params[model_name]
    side_loss = mparams.get("pool_loss", False)
    if not mol_filter or 'gcn_laplacian' not in model_name:
        mol_filter = mparams.get('pool_arch', {}).get('hidden_dim', 0)

    try :
        min_num_nodes = int(np.ceil(mparams['pool_arch'].get('hidden_dim')*max_num_nodes))
    except:
        min_num_nodes = int(config.pop('min_num_nodes', 0))


    if 'gcn_laplacian' in model_name:
        reg_mode = int(config.get("reg_mode", 1))
        hop = int(config.get("hop", 3))
        lap_hop = int(config.get("lap_hop", 1))
        init_params = dict(reg_mode=reg_mode, hop=hop, lap_hop=lap_hop)
        mparams['pool_arch'].update(init_params)

    else:
        if mparams.get("pool_arch"):
            mparams['pool_arch'].update({'hidden_dim': min_num_nodes})

    # Set up data
    import time
    start = time.time()
    num_output = None
    if graph_type == 'dense':
        X, y = read_dense_graph_data(input_path, min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes)
        num_output = len(set(y))
    else:
        raise NotImplemented

    # X, y, *meta = read_data(input_path, n_out, format=file_type, max_n=max_n)
    print(mparams)
    out_dim = num_output if num_output > 2 else 1
    print(out_dim)
    end = time.time()
    print("Data processing took {:.2f} seconds".format(end - start))

    """
    scname = model_params.pop("yscaler", "none")
    y_scaler = get_scaler(scname)
    # Transform y given the scaler
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    """

    binary = out_dim == 1

    METRICS = {'acc': partial(score_fn, scorer=accuracy, binary=binary), 'roc': partial(score_fn, scorer=roc_auc_score, auc=True, binary=binary)}
    EVAL_METRICS = {'acc': partial(score_fn, scorer=accuracy, binary=binary), 'roc': partial(score_fn, scorer=roc_auc_score, auc=True, binary=binary),
                    'f1_macro': partial(score_fn, scorer=f1_score, binary=binary, average='macro')}

    eval_metrics = EVAL_METRICS.keys()
    metrics_results = defaultdict(list)
    results = []
    model = Encoder
    batch_size = 16
    test_batch_size = 16
    base_loss = nn.BCEWithLogitsLoss(reduction='sum') if binary else nn.CrossEntropyLoss(reduction="sum")
    train_generator = partial(batch_generator_sup, balance=balance, shuffle=True)
    test_generator = partial(batch_generator_sup, infinite=False, shuffle=False, balance=balance)
    
    for repeat in range(repeats):
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed += 1

        train_dt, valid_dt, test_dt = load_mol_dataset(X, y, test_size=test_size, device=device,
                                                   valid_size=valid_size,
                                                   shuffle=True, balance=balance,
                                                   with_bond=with_bond, mol_filter=mol_filter, type=graph_type,
                                                   stratify=stratify)

        step_train = int(np.ceil(len(train_dt) / batch_size))
        step_valid = int(np.ceil(len(valid_dt) / batch_size))
        step_test = int(np.ceil(len(test_dt) / test_batch_size))

        in_size = train_dt[0][0][1].shape[-1]
        nedges = 1 #train_dt[0][0][0].shape[-1]  # can be either 1 or n_edges


        clf = model(input_dim=in_size, out_dim=out_dim, nedges=nedges, **mparams)
        loss_fn = partial(fingerprint_multitask_loss, model=clf, base_criterion=base_loss, side_loss=side_loss, binary=binary)

        trainer = SupervisedTrainer(clf, loss_fn=loss_fn, metrics=METRICS, gpu=(device == "cuda"),
                                    model_dir=output_path, op__lr=1e-4)

        trainer.fit(train_dt, valid_dt, batch_size=batch_size, epochs=num_epochs, generator_fn=train_generator,
                    steps_per_epoch=step_train, validation_steps=step_valid, checkpoint='model.pth.tar', tboardX="logs",
                    checkpoint__restore_best=True, reduce_lr={"verbose":True, "factor":0.5, "cooldown":5}, early_stopping={"patience":50, "min_delta":1e-6})
                 
        res = trainer.evaluate(test_dt, batch_size=test_batch_size, return_pred=True, generator_fn=test_generator,
                               steps=step_test)
        if binary:
            y_pred = torch.sigmoid(torch.Tensor(res[2])).numpy()
        else:
            y_pred = torch.softmax(torch.Tensor(res[2]), dim=-1).numpy()

        for metric_name, metric in EVAL_METRICS.items():

            # IF ROC, recover from error using macro average over tasks, with 0.5 task score as default when undefined.
            if metric_name == 'roc':
                try:
                    m_value = metric(y_pred, test_dt.y)
                except ValueError as e:
                    print(e)
                    if len(test_dt.y.shape) == 1:
                        m_value = 0.5
                    elif len(test_dt.y.shape) == 2:
                        scores = []
                        for col in range(test_dt.y.shape[1]):
                            if len(np.unique(test_dt.y[:, col])) == 1:
                                scores.append(0.5)
                            else:
                                scores.append(metric(y_pred[:, col], test_dt.y[:, col]))
                        m_value = np.asarray(scores).mean()
                    else:
                        raise e
            else:
                m_value = metric(y_pred, test_dt.y)
            metrics_results[metric_name].append(m_value)
        print(["{} {}".format(m , metrics_results[m][-1]) for m in eval_metrics])
        np.savez_compressed(os.path.join(output_path, 'test_{}_{}.npz'.format(model_name, repeat)),
                            test=test_dt.y,
                            pred=y_pred)
        save_model(clf, os.path.join(output_path, "model:{}_{}.pkl".format(model_name, repeat)))

    tmp_results = dict([('name', model_name), ('repeats', repeats)])
    for name, vals in metrics_results.items():
        tmp_results[name] = vals
    print(tmp_results)
    results.append(tmp_results)
    mparams['data'] = config['data_path']
    with open(os.path.join(output_path, "config_{}.pkl".format(model_name)), 'wb') as out_file:
        pickle.dump(mparams, out_file, protocol=4)
    with open(os.path.join(output_path, "data_{}.pkl".format(model_name)), 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)
