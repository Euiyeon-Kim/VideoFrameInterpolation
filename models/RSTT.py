import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.losses import *
from models.IFRNet import Encoder
from models.IFRM2M import Decoder4v1
from modules.warp import bwarp, fwarp
from models.GMTrans import Decoder2
from utils.flow_viz import flow_tensor_to_np


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, recompute_scale_factor=False,
                         mode="bilinear", align_corners=True)


class RSTTv1(nn.Module):
    def __init__(self, args):
        super(RSTTv1, self).__init__()
        self.args = args

        self.encoder = Encoder()
        self.decoder4 = Decoder4v1()
        self.query_builder3 = nn.Conv2d(72 * 2, 72, 3, 1, 1)
        self.decoder3 = Decoder2(inp_dim=72, depth=6, num_heads=6, window_size=4)

        self.query_builder2 = nn.ConvTranspose2d(72, 48, 4, 2, 1)
        self.decoder2 = Decoder2(inp_dim=48, depth=6, num_heads=6, window_size=4)

        self.query_builder1 = nn.ConvTranspose2d(48, 32, 4, 2, 1)
        self.decoder1 = Decoder2(inp_dim=32, depth=4, num_heads=4, window_size=2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upconv1 = nn.Conv2d(32, 32 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_last = nn.Conv2d(32, 3, 3, 1, 1)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def generate_rgb_frame(self, feat, m):
        out = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        return torch.clamp(out + m, 0, 1)

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1 = x0 / 255., x1 / 255.
        mean_ = (torch.mean(x0, dim=(2, 3), keepdim=True) + torch.mean(x1, dim=(2, 3), keepdim=True)) / 2
        x0_, x1_ = x0 - mean_, x1 - mean_

        feat0_1, feat0_2, feat0_3, feat0_4 = self.encoder(x0_)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.encoder(x1_)

        # Predict coarsest flow
        out4 = self.decoder4(feat0_4, feat1_4)
        f01_4, f10_4 = out4[:, 0:2], out4[:, 2:4]

        # Calculate Z
        x0_4, x1_4 = resize(x0_, scale_factor=1 / 8), resize(x1_, scale_factor=1 / 8)
        z0_4 = (1.0 - (x0_4 - bwarp(x1_4, f01_4)).abs().mean([1], True)).clip(0.001, None).square()
        z1_4 = (1.0 - (x1_4 - bwarp(x0_4, f10_4)).abs().mean([1], True)).clip(0.001, None).square()

        # Build first query
        ft0_3 = - fwarp(f01_4, f01_4 * t, z0_4) * t
        ft1_3 = - fwarp(f10_4, f01_4 * (1 - t), z1_4) * (1 - t)
        feat_t_3_from_feat_0_3 = bwarp(feat0_3, ft0_3)
        feat_t_3_from_feat_1_3 = bwarp(feat1_3, ft1_3)
        feat_t_3_query = self.query_builder3(torch.cat((feat_t_3_from_feat_0_3, feat_t_3_from_feat_1_3), dim=1))

        # Attention
        pred_feat_t_3 = self.decoder3(feat_t_3_query, feat0_3, feat1_3)

        feat_t_2_query = self.query_builder2(pred_feat_t_3)
        pred_feat_t_2 = self.decoder2(feat_t_2_query, feat0_2, feat1_2)

        feat_t_1_query = self.query_builder1(pred_feat_t_2)
        pred_feat_t_1 = self.decoder1(feat_t_1_query, feat0_1, feat1_1)

        imgt_pred = self.generate_rgb_frame(pred_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        f01, f10 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        xt_ = xt - mean_
        ft_1, ft_2, ft_3, ft_4 = self.encoder(xt_)

        x0_pred = self.generate_rgb_frame(feat0_1, mean_)
        x1_pred = self.generate_rgb_frame(feat1_1, mean_)

        l1_loss_inter = self.l1_loss(imgt_pred - xt)
        l1_loss_origin = self.l1_loss(x0_pred - x0) + self.l1_loss(x1_pred - x1)
        l1_loss = l1_loss_origin + l1_loss_inter

        census_loss_inter = self.tr_loss(imgt_pred, xt)
        census_loss_origin = self.tr_loss(x0_pred, x0) + self.tr_loss(x1_pred, x1)
        census_loss = census_loss_origin + census_loss_inter

        geo_loss = 0.01 * (self.gc_loss(pred_feat_t_1, ft_1) +
                           self.gc_loss(pred_feat_t_2, ft_2) +
                           self.gc_loss(pred_feat_t_3, ft_3))
        pred_f01, pred_f10 = resize(f01_4, 8.0) * 8.0, resize(f10_4, 8.0) * 8.0
        robust_weight0 = get_robust_weight(pred_f01, f01, beta=0.3)
        robust_weight1 = get_robust_weight(pred_f10, f10, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(pred_f01 - f01, weight=robust_weight0) +
                               self.rb_loss(pred_f10 - f10, weight=robust_weight1))
        total_loss = l1_loss + census_loss + geo_loss + distill_loss

        return {
            'frame_preds': [imgt_pred],
            'f01': pred_f01,
            'f10': pred_f10,
            'origin_preds': [x0_pred, x1_pred],
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'l1_inter': l1_loss_inter.item(),
            'l1_origin': l1_loss_inter.item(),
            'census_loss': census_loss.item(),
            'census_inter': census_loss_inter.item(),
            'census_origin': census_loss_origin.item(),
            'flow_loss': distill_loss.item(),
            'geometry_loss': geo_loss.item(),
        }

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0]/255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
        viz_flow = torch.cat((x0_01, torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), x1_01), dim=-1)
        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        pred_x0 = results_dict['origin_preds'][0][0][None]
        pred_x1 = results_dict['origin_preds'][-1][0][None]
        x0_err_map = (x0_01 - pred_x0).abs()
        x1_err_map = (x1_01 - pred_x1).abs()
        process_concat = torch.cat((pred_x0, x0_err_map, pred_x1, x1_err_map), dim=-1)

        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
            'process': process_concat[0],
        }
