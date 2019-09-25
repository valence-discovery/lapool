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
using the --arch flag. 

The default architecture is specified by the -c flag, while any additional parameters 
to use during training can be specified via the -h flag and will override the previous arguments.

An example command to run on each dataset using
GIN benchnmark would thus be:

python supervised.py -d 'tox21' -a 'gnn' -c 'default_config.yml'

python supervised.py -d 'alerts' -a 'gnn' -c 'default_config.yml'

python supervised.py -d 'fragments' -a 'gnn' -c 'default_config.yml'

### Generative experiments
