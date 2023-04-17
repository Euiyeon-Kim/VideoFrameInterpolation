import os
import random

import numpy as np
import oyaml as yaml
from dotmap import DotMap

import torch
import torch.distributed as dist


def prepare_env(args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True


def get_options(parsed):
    with open(parsed.config, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
    args.config = parsed.config
    args.exp_name = parsed.exp_name
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.log_dir = os.path.join('exps', args.exp_name)
    args.resume = parsed.resume                 # Todo: Implement resume
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


