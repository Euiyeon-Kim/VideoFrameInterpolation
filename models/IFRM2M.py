import torch
import torch.nn as nn

from modules.warp import bwarp
from models.IFRNet import Encoder, ResBlock, convrelu, resize
from modules.losses import Ternary, Geometry, Charbonnier_Ada, Charbonnier_L1


class Decoder4v1(nn.Module):
    """
        Get two features and calculate bi-directional flow
    """
    def __init__(self, nc=96):
        super(Decoder4v1, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            convrelu(nc * 2, nc * 2),
            ResBlock(nc * 2, 32),
            nn.ConvTranspose2d(nc * 2, 4, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1):
        f_in = torch.cat([f0, f1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder32v1(nn.Module):
    """
        Input: source, source_bwarp_target, prev level importance score
        Output: multiple flow source to target and source importance score
    """
    def __init__(self, nc=72):
        super(Decoder32v1, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            convrelu(nc * 2 + 1, nc * 2),
            ResBlock(nc * 2, 32),
            nn.ConvTranspose2d(nc * 2, 2 + 1, 4, 2, 1, bias=True)
        )

    def forward(self, feat0, feat1, f01, f10, z0, z1):
        b, _, h, w = feat0.shape

        def _process(source, target, f_st, z):
            source_bwarp_target = warp(target, f_st)
            f_in = torch.cat([source, source_bwarp_target, z], 1)
            f_out = self.convblock(f_in)
            res_flow = f_out[:, :2 * self.n_branch].view(b, self.n_branch, 2, h * 2, w * 2)
            res_z = torch.sigmoid(f_out[:, 2 * self.n_branch:2 * self.n_branch + 1]) * 0.8 + 0.1
            return res_flow, res_z

        res_f01, res_z0 = _process(feat0, feat1, f01, z0)
        res_f10, res_z1 = _process(feat1, feat0, f10, z1)

        return res_f01, res_f10, res_z0, res_z1


class IFRM2Mv1(nn.Module):
    def __init__(self, args):
        super(IFRM2Mv1, self).__init__()
        self.args = args
        self.n_branch = args.m2m_branch
        self.distill_lambda = args.distill_lambda
        self.alpha = torch.nn.Parameter(10.0 * torch.ones(1, 1, 1, 1))

        self.encoder = Encoder()
        self.decoder4 = Decoder4v1(nc=96)
        # self.decoder3 = Decoder3(n_branch=self.n_branch)
        # self.decoder2 = Decoder21(nc=48, n_branch=self.n_branch)
        # self.decoder1 = Decoder21(nc=32, n_branch=self.n_branch)

        self.tr_loss = Ternary(7)
        self.gc_loss = Geometry(3)
        self.l1_loss = Charbonnier_L1()
        self.rb_loss = Charbonnier_Ada()

    def forward(self, inp_dict):
        '''
            img0, img1: Normalized [-1, 1]
            embt: [B, 1, 1, 1]
        '''
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']
        b, _, h, w = x0.shape

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1, xt = x0 / 255., x1 / 255., xt / 255
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_

        # Generate multi-level features
        feat0_1, feat0_2, feat0_3, feat0_4 = self.encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.encoder(x1)

        # Process decoder4 output
        out4 = self.decoder4(feat0_4, feat1_4)
        f01_4, f10_4 = out4[:, 0:2], out4[:, 2:4]
        z0 = bwarp(x1, resize(f01_4, scale_factor=8) * 8)
        print(z0.shape)
        exit()
