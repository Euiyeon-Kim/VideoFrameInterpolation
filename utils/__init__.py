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
