import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.img_dict = {}      # For logging

    @staticmethod
    def norm_w_rgb_mean(x0, x1):
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_
        return x0, x1, mean_

    @staticmethod
    def resize(x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor,
                             recompute_scale_factor=False, mode="bilinear", align_corners=True)

    @abstractmethod
    def get_img_dict(self, *args, **kwargs):
        return

    @abstractmethod
    def inference(self, *args, **kwargs):
        return

    @abstractmethod
    def forward(self, *args, **kwargs):
        return
