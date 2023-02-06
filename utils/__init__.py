import random
import shutil

import numpy as np
import oyaml as yaml
from dotmap import DotMap

import torch


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_yaml(exp_name, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    shutil.copy(config_path, f"exps/{exp_name}/config.yaml")
    args = DotMap(config)
    return args
