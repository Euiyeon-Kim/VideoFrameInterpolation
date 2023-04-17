import torch
import torch.nn as nn

from .dcnv2 import DeformableConv2d


class DCNInterFeatBuilderwithT(nn.Module):
    """
        Backward warping으로 query building
    """
    def __init__(self, nc):
        super(DCNInterFeatBuilderwithT, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            nn.Conv2d(nc * 2 + 1, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.PReLU(nc),
        )
        self.dcnt0 = DeformableConv2d(nc, nc)
        self.dcnt1 = DeformableConv2d(nc, nc)
        self.blendblock = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
        )

    def forward(self, feat0, feat1, t):
        _, c, fh, fw = feat0.shape
        concat_t = t.repeat(1, 1, fh, fw)
        f01_motion_feat = self.convblock(torch.cat((feat0, feat1, concat_t), 1))
        f10_motion_feat = self.convblock(torch.cat((feat1, feat0, 1 - concat_t), 1))
        feat_t_from_feat0, ft0_offset = self.dcnt0(feat0, f01_motion_feat)
        feat_t_from_feat1, ft1_offset = self.dcnt1(feat1, f10_motion_feat)
        out = self.blendblock(torch.cat((feat_t_from_feat0, feat_t_from_feat1), 1))
        return out, ft0_offset, ft1_offset
