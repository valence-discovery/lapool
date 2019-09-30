import click
import json
import os
import torch
import numpy as np
import warnings
import torch.nn as nn
import pickle
import sys
import tarfile
import gc
import yaml

warnings.filterwarnings(action='ignore')
from functools import partial
from collections import defaultdict
from gnnpooling.utils.read_data import read_gen_data
from gnnpooling.utils.trainer import AAETrainer, GANScheduler, TrainerCheckpoint, EarlyStopping
from gnnpooling.models.networks import Encoder, Decoder, MLPdiscriminator
from gnnpooling.models.aae import AAE
from gnnpooling.utils.datasets import GraphDataLoader
from tensorboardX import SummaryWriter
from pytoune.framework.callbacks import TensorBoardLogger
from joblib import Parallel, delayed
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def save_model(model, output):
    try:
        model.save(output)
    except:
        joblib.dump(model, output)

@click.command()
@click.option('--arch', '-a', default='gnn', help="Type of model")
@click.option('--dataset', '-d', required=True, help="Path to dataset")
@click.option('--max_nodes', default=9, type=int, help="Maximum number of nodes")
@click.option('--min_nodes', default=0, type=int, help="Minimum number of nodes")
@click.option('--ksize', '-k', default=0.10, type=float, help="Percentage of nodes to retains during hierarchical pooling")
@click.option('--config_file', '-c', required=True, type=click.Path(exists=True), help="File containing the model global configuration file")
@click.option('--hparams', '-h', type=click.Path(exists=True), help="File containing the hpool params")
@click.option('--output_path', '-o', default="", help="Output folder")
@click.option('--epochs', '-e', type=int, default=100, help="Number of epochs")
@click.option('--batch_size', '-b', type=int, default=32, help="Batch size")
@click.option('--max_n', default=-1, type=int, help="Maxinum number of datum to consider")
@click.option('--cpu', is_flag=True, help="Force use cpu")
@click.option('--samples', default=1000, type=int,  help="Number of molecules to samples")
@click.option('--save_every', default=0.25, type=float,  help="Percentage of saves per epochs")
def cli(arch, dataset, max_nodes, min_nodes, ksize, config_file, hparams, output_path, epochs, batch_size, max_n, cpu, samples, save_every):
    torch.backends.cudnn.benchmark = True
    np.random.seed(42)
    torch.manual_seed(42)
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
    nedges = 3 #
    hpool_params.update(hpool_params_add)
    print(hpool_params)
    config['encoder']["hpool"].update(hpool_params)
    train_dt, test_dt, valid_dt, n_embed, atom_list = read_gen_data(dataset, max_n=max_n, min_size=min_nodes, max_size=max_nodes, all_feat=False, add_bond=(nedges >1), one_hot_bond=True)
    # X, y, mols = transform_data(smiles, y, min_size=min_size, max_size=max_size, all_feat=all_feat, add_bond=with_bond)
    output_path = os.path.join(output_path, arch)
    latent_dim = int(config.get("latent_dim",  512))
    train_generator = GraphDataLoader(train_dt, batch_size=batch_size, shuffle=True)
    valid_generator = GraphDataLoader(valid_dt, batch_size=batch_size, shuffle=True)
    callback = []
    tboardX = {"log_dir" : os.path.join(output_path, "logs")}
    step_per_epochs = len(train_dt) // batch_size
    save_every = int(np.ceil(step_per_epochs*save_every))
    writer = SummaryWriter(**tboardX) 
    checkpoint = os.path.join(output_path, 'model.epoch:{epoch:02d}-loss:{loss:.2f}.pth.tar')
    checkpoint_params = dict(save_best_only=True, temporary_filename=checkpoint + ".tmp")
    check_callback = TrainerCheckpoint(checkpoint, **checkpoint_params)
    callback.append(check_callback)
    callback.append(GANScheduler(torch_lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, optim="gen", factor=0.5, verbose=True, patience=50))
    #callback.append(EarlyStopping(patience=20))
    callback.append(TensorBoardLogger(writer))
    
    network = AAE(atom_decoder=atom_list+['*'], max_size=max_nodes, nedges=nedges+1)
    encoder = Encoder(n_embed, latent_dim, nedges+1, **config["encoder"])
    decoder = Decoder(latent_dim, n_embed, max_nodes, nedges+1, **config["decoder"])
    discriminator = MLPdiscriminator(latent_dim, **config["discriminator"])
    
    network.set_encoder(encoder)
    network.set_decoder(decoder)
    network.set_discriminator(discriminator)

    trainer = AAETrainer(network, loss_fn=None, metrics={},  gpu=(not cpu), D_op__lr=5e-3, G_op__lr=1e-2, model_dir=output_path, save_every=save_every) 
    trainer.writer = writer
    trainer.set_training_mode(pretrain=False)
    trainer.fit_generator(train_generator, valid_generator, epochs=epochs, callbacks=callback)

    mol_generated = trainer.sample_molecules(samples)
    with open(os.path.join(output_path, "mols.txt"), "w") as MOL_OUT:
        MOL_OUT.write("\n".join(mol_generated))
    del mol_generated
    gc.collect()
    test_generator = GraphDataLoader(test_dt, batch_size=batch_size)
    pred_dict = trainer.predict_generator(test_generator)
    with open(os.path.join(output_path, "predictions.json"), 'w') as out_file:
        json.dump(pred_dict, out_file)
    #print(pred_dict)

if __name__ == '__main__':
    cli()
