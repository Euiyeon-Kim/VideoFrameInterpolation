import torch
import torch.nn.functional as F

from modules.softsplat import softsplat_func


def bwarp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def fwarp(tenIn:torch.Tensor, tenFlow:torch.Tensor, tenMetric:torch.Tensor, strMode:str='soft'):
    assert(strMode.split('-')[0] in ['sum', 'avg', 'linear', 'soft'])

    if strMode == 'sum': assert(tenMetric is None)
    if strMode == 'avg': assert(tenMetric is None)
    if strMode.split('-')[0] == 'linear': assert(tenMetric is not None)
    if strMode.split('-')[0] == 'soft': assert(tenMetric is not None)

    if strMode == 'avg':
        tenIn = torch.cat([tenIn, tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]])], 1)

    elif strMode.split('-')[0] == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split('-')[0] == 'soft':
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if strMode.split('-')[0] in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split('-')) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'addeps':
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'zeroeps':
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split('-')[1] == 'clipeps':
            tenNormalize = tenNormalize.clip(0.0000001, None)

        # end

        tenOut = tenOut[:, :-1, :, :] / tenNormalize
    # end

    return tenOut


"""
    fwarp_using_two_frames and fwarp_mframes is originated from 
    https://github.com/feinanshan/M2M_VFI/blob/main/Train/vfi/model/m2m.py
"""


def fwarp_using_two_frames(tenIn1, tenFlow1, t1, tenIn2, tenFlow2, t2, tenMetric1=None, tenMetric2=None):
    def one_fdir(tenIn, tenFlow, td, tenMetric):
        tenIn = torch.cat([tenIn * td * (tenMetric).clip(-20.0, 20.0).exp(),
                           td * (tenMetric).clip(-20.0, 20.0).exp()], 1)
        tenOut = softsplat_func.apply(tenIn, tenFlow)
        return tenOut[:, :-1, :, :], tenOut[:, -1:, :, :] + 0.0000001
    # end

    tenOutF, tenNormalizeF = one_fdir(tenIn1, tenFlow1, t1, tenMetric1)
    tenOutB, tenNormalizeB = one_fdir(tenIn2, tenFlow2, t2, tenMetric2)

    tenOut = tenOutF + tenOutB
    tenNormalize = tenNormalizeF + tenNormalizeB
    # end

    return tenOut / tenNormalize, tenNormalize < 0.00001


def fwarp_mframes(tenIn1, tenFlow1, t1, tenIn2, tenFlow2, t2, tenMetric1, tenMetric2):
    def repeat_for_branch(tenIn, nb):
        _b, _c, _h, _w = tenIn.shape
        return tenIn.repeat(1, nb, 1, 1).view(_b, nb, _c, _h, _w).permute(1, 0, 2, 3, 4)

    def one_fdir(tenIn, tenFlow, td, tenMetric):
        tenIn = torch.cat([tenIn * td * (tenMetric).clip(-20.0, 20.0).exp(),
                           td * (tenMetric).clip(-20.0, 20.0).exp()], 1)
        tenOut = softsplat_func.apply(tenIn, tenFlow)

        return tenOut[:, :-1, :, :], tenOut[:, -1:, :, :] + 0.0000001

    n_branch = tenFlow1.shape[0]
    tenIn1 = repeat_for_branch(tenIn1, n_branch)
    tenIn2 = repeat_for_branch(tenIn2, n_branch)
    tenMetric1 = repeat_for_branch(tenMetric1, n_branch)
    tenMetric2 = repeat_for_branch(tenMetric2, n_branch)

    tenOut = 0
    tenNormalize = 0
    for idx in range(n_branch):
        tenOutF, tenNormalizeF = one_fdir(tenIn1[idx], tenFlow1[idx], t1[idx], tenMetric1[idx])
        tenOutB, tenNormalizeB = one_fdir(tenIn2[idx], tenFlow2[idx], t2[idx], tenMetric2[idx])

        tenOut += tenOutF + tenOutB
        tenNormalize += tenNormalizeF + tenNormalizeB
    # end

    return tenOut / tenNormalize, tenNormalize < 0.00001