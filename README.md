# Directory contents

supervised.py - main run script for supervised experiments 

default_config.yml - default architecture configuration

bin/ - executables for running experiments

data/  - Source datasets

expts/ - configuration files

gnnpooling/ - main source code with pooling and gnn modules

results/ - experiment results


# Running experiments

### Supervised experiments
Running the supervised experiments can be done using the train_supervised script located in bin/.
Specifying the experiment to run on is done using the --dataset flag. The GNN/pooling algorithm to use is specified

The default architecture is specified by the -c flag, while any additional parameters to use during training can be specified via the -h flag and will override the previous arguments.

An example command to run lapool on the tox21 dataset would thus be:

`python supervised.py -c default_config.yml  --dataset 'tox21' -o output  -k 0 -e 100`

This will run lapool (default model) using the automatic centroid detection (k=0), default parameters, a maximum number of 100 epochs (early stopping is used in this sample code).

You should expect a roc-auc of about 0.815 on the test set.

```
{
'valid': {'acc': 0.733116014710799, 'roc': array([0.77105858, 0.82134402, 0.85963513, 0.78382353, 0.71445147,
       0.83046739, 0.81030658, 0.73230864, 0.83205791, 0.83140754,
       0.84894438, 0.84327084]), 'roc_macro': 0.8065896671183973}, 

'test': {'acc': 0.7528843155597131, 'roc': array([0.76190476, 0.87559682, 0.85369492, 0.83261082, 0.70596987,
       0.82577614, 0.85762631, 0.75853388, 0.84970845, 0.77274431,
       0.86176906, 0.82456763]), 'roc_macro': 0.8150419140496449}
}
```

### Generative experiments

Example of running with default qm9 dataset (0-9 atoms), on 10 epochs.

`python generative.py -d qm9 -c expts/aae/1.yaml -o test  -e 10
