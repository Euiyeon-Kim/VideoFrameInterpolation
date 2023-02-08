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


def normalize_imgnet(frames):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(frames.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(frames.device)
    frames = (frames / 255. - mean) / std
    return frames


def denormalize_imgnet_to01(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
    return (img_tensor * std) + mean

