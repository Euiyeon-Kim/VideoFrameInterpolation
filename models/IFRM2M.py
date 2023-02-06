import torch
import torch.nn as nn

from modules.warp import bwarp, fwarp_mframes
from models.IFRNet import Encoder, ResBlock, convrelu, resize
from modules.losses import Ternary, Geometry, Charbonnier_Ada, Charbonnier_L1, get_robust_weight


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

        def _process(source, target, f_st, z_s):
            source_bwarp_target = bwarp(target, f_st)
            f_in = torch.cat([source, source_bwarp_target, z_s], 1)
            f_out = self.convblock(f_in)
            res_flow = f_out[:, :2]
            res_z = torch.sigmoid(f_out[:, 2:]) * 0.99 + 0.01
            return res_flow, res_z

        res_f01, res_z0 = _process(feat0, feat1, f01, z0)
        res_f10, res_z1 = _process(feat1, feat0, f10, z1)

        return res_f01, res_f10, res_z0, res_z1


class Decoder1v1(nn.Module):
    def __init__(self, nc=32, n_branch=9):
        super(Decoder1v1, self).__init__()
        self.nc = nc
        self.n_branch = n_branch
        self.convblock = nn.Sequential(
            convrelu(nc * 2 + 1, nc * 2),
            ResBlock(nc * 2, 32),
            nn.ConvTranspose2d(nc * 2, 2 * n_branch + 1, 4, 2, 1, bias=True)
        )

    def forward(self, feat0, feat1, f01, f10, z0, z1):
        b, _, h, w = feat0.shape

        def _process(source, target, f_st, z_s):
            source_bwarp_target = bwarp(target, f_st)
            f_in = torch.cat([source, source_bwarp_target, z_s], 1)
            f_out = self.convblock(f_in)
            res_flow = f_out[:, :2 * self.n_branch].view(b, self.n_branch, 2, h * 2, w * 2)
            res_z = torch.sigmoid(f_out[:, 2 * self.n_branch:2 * self.n_branch + 1]) * 0.99 + 0.01
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
        self.decoder3 = Decoder32v1(nc=72)
        self.decoder2 = Decoder32v1(nc=48)
        self.decoder1 = Decoder1v1(nc=32, n_branch=self.n_branch)

        self.tr_loss = Ternary(7)
        self.gc_loss = Geometry(3)
        self.l1_loss = Charbonnier_L1()
        self.rb_loss = Charbonnier_Ada()

    def repeat_for_branch(self, tensor):
        _b, _c, _h, _w = tensor.shape
        return tensor.repeat(1, self.n_branch, 1, 1).view(_b, self.n_branch, _c, _h, _w).permute(1, 0, 2, 3, 4)

    def forward(self, inp_dict):
        '''
            img0, img1: Normalized [-1, 1]
            embt: [B, 1, 1, 1]
        '''
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']
        b, _, h, w = x0.shape

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        nt = self.repeat_for_branch(t)
        x0, x1, xt = x0 / 255., x1 / 255., xt / 255
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_

        # Generate multi-level features
        feat0_1, feat0_2, feat0_3, feat0_4 = self.encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.encoder(x1)

        # Process decoder4 output
        out4 = self.decoder4(feat0_4, feat1_4)
        f01_4, f10_4 = out4[:, 0:2], out4[:, 2:4]
        x0_4 = resize(x0, scale_factor=1 / 8)
        x1_4 = resize(x1, scale_factor=1 / 8)
        z0_4 = (1.0 - (x0_4 - bwarp(x1_4, f01_4)).abs().mean([1], True)).clip(0.001, None).square()
        z1_4 = (1.0 - (x1_4 - bwarp(x0_4, f10_4)).abs().mean([1], True)).clip(0.001, None).square()

        f01_res_3, f10_res_3, z0_res_3, z1_res_3 = self.decoder3(feat0_3, feat1_3, f01_4, f10_4, z0_4, z1_4)
        f01_3 = 2.0 * resize(f01_4, scale_factor=2.0) + f01_res_3
        f10_3 = 2.0 * resize(f10_4, scale_factor=2.0) + f10_res_3
        z0_3, z1_3 = z0_res_3 + resize(z0_4, 2.0), z1_res_3 + resize(z1_4, 2.0)

        f01_res_2, f10_res_2, z0_res_2, z1_res_2 = self.decoder2(feat0_2, feat1_2, f01_3, f10_3, z0_3, z1_3)
        f01_2 = 2.0 * resize(f01_3, scale_factor=2.0) + f01_res_2
        f10_2 = 2.0 * resize(f10_3, scale_factor=2.0) + f10_res_2
        z0_2, z1_2 = z0_res_2 + resize(z0_3, 2.0), z1_res_2 + resize(z1_3, 2.0)

        f01_res_1, f10_res_1, z0_res_1, z1_res_1 = self.decoder1(feat0_1, feat1_1, f01_2, f10_2, z0_2, z1_2)
        f01_1 = (2.0 * resize(f01_2, scale_factor=2.0).unsqueeze(1) + f01_res_1).permute(1, 0, 2, 3, 4)
        f10_1 = (2.0 * resize(f10_2, scale_factor=2.0).unsqueeze(1) + f10_res_1).permute(1, 0, 2, 3, 4)
        z0_1, z1_1 = z0_res_1 + resize(z0_2, 2.0), z1_res_1 + resize(z1_2, 2.0)

        pred_xt, is_blank = fwarp_mframes(x0, f01_1 * nt, nt, x1, f10_1 * (1 - nt), (1 - nt),
                                          z0_1 * self.alpha, z1_1 * self.alpha)
        for_blank = (((1 - t) * x0 + t * x1) * is_blank * 1.0)
        imgt_pred = torch.clamp(for_blank + pred_xt + mean_, 0, 1)

        if not self.training:
            return imgt_pred

        # Calculate loss
        f01, f10 = inp_dict['f01'], inp_dict['f10']
        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)

        mu_f01_1, mu_f10_1 = torch.mean(f01_1, 0), torch.mean(f10_1, 0)
        robust_weight0 = get_robust_weight(mu_f01_1, f01, beta=0.3)
        robust_weight1 = get_robust_weight(mu_f01_1, f10, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(2.0 * resize(f01_2, 2.0) - f01, weight=robust_weight0) +
                               self.rb_loss(2.0 * resize(f10_2, 2.0) - f10, weight=robust_weight1) +
                               self.rb_loss(4.0 * resize(f01_3, 4.0) - f01, weight=robust_weight0) +
                               self.rb_loss(4.0 * resize(f10_3, 4.0) - f10, weight=robust_weight1) +
                               self.rb_loss(8.0 * resize(f01_4, 8.0) - f01, weight=robust_weight0) +
                               self.rb_loss(8.0 * resize(f10_4, 8.0) - f10, weight=robust_weight1))
        total_loss = l1_loss + census_loss + distill_loss

        return {
            'frame_preds': [imgt_pred],
            'xt_warp_x0': for_blank + mean_,
            'xt_warp_x1': pred_xt + mean_,
            'x0_mask': z0_1,
            'f01': mu_f01_1,
            'f10': mu_f10_1,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'flow_loss': distill_loss.item(),
            'geometry_loss': 0.0,
            'alpha': self.alpha[0].item()
        }
