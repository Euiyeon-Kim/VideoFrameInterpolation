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