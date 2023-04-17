import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor,
                         recompute_scale_factor=False, mode="bilinear", align_corners=True)


def normalize_imgnet(frames):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(frames.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(frames.device)
    frames = (frames / 255. - mean) / std
    return frames


def denormalize_imgnet_to01(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
    return (img_tensor * std) + mean


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
