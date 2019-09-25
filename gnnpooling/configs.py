import glob
import os
import yaml
basedir = os.path.dirname(__file__)


def gen_model_getter(arch, model_id):
    path = os.path.join(basedir, "../expts/", arch.lower(), str(model_id)+"*")
    with open(glob.glob(path)[0], 'r') as IN:
        model_params = yaml.safe_load(IN)
    return model_params
 

def sup_model_getter(model_id):
    path = os.path.join(basedir, "../expts/supervised", str(model_id)+"*")
    with open(glob.glob(path)[0], 'r') as IN:
        model_params = yaml.safe_load(IN)
    return model_params     