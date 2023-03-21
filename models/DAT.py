import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.warp import bwarp
from models.base import Basemodel
from models.IFRNet import convrelu, ResBlock
from utils.flow_viz import flow_tensor_to_np
from modules.deformable_attn import DeformAttn
from modules.residual_encoder import make_layer
from models.DCNTrans import DCNInterFeatBuilder
from modules.positional_encoding import PositionEmbeddingSine
from modules.losses import Ternary, Geometry, Charbonnier_Ada, Charbonnier_L1, get_robust_weight


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor,
                         recompute_scale_factor=False, mode="bilinear", align_corners=True)


class ResEncoder(nn.Module):
    def __init__(self, nf, n_res_block):
        super(ResEncoder, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
        )
        self.res_blocks = make_layer(nf=nf, n_layers=n_res_block)
        self.fea_L2_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.PReLU(nf),
        )
        self.fea_L3_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.PReLU(nf),
        )
        self.fea_L4_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.PReLU(nf),
        )

    def forward(self, x):
        feat1 = self.res_blocks(self.projection(x))
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        feat4 = self.fea_L4_conv(feat3)
        return feat1, feat2, feat3, feat4


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        return out


class CrossDeformableAttentionBlockwFlow(nn.Module):
    def __init__(self, in_c, out_c, n_samples=9, n_groups=12, n_heads=12, mlp_ratio=2.0, pred_res_flow=True):
        super(CrossDeformableAttentionBlockwFlow, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.n_groups = n_groups
        self.n_group_c = in_c // n_groups
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.pred_res_flow = pred_res_flow
        self.conv_res_feat = nn.Sequential(
            convrelu(self.in_c * 2 + 2, self.in_c * 2),
            convrelu(self.in_c * 2, self.in_c),
            ResBlock(self.in_c, self.in_c // 2)
        )
        # Calculate residual offset
        self.conv_res_offset = nn.Conv2d(self.in_c, n_groups * n_samples * 2, 3, 1, 1)
        self.conv_res_offset.apply(self._init_zero)
        if pred_res_flow:
            self.conv_res_flow = nn.ConvTranspose2d(self.in_c, 2, 4, 2, 1, bias=True)

        self.attn = DeformAttn(in_c, out_c, n_samples, n_groups, n_heads)
        self.mlp = Mlp(in_features=out_c, hidden_features=out_c * mlp_ratio, out_features=out_c)
        self.blendblock = nn.Sequential(
            convrelu(self.out_c * 2, self.out_c),
            convrelu(self.out_c, self.out_c),
        )

    @staticmethod
    def _init_zero(m):
        if type(m) == nn.Conv2d:
            m.weight.data.zero_()
            m.bias.data.zero_()

    def _get_movement_feats(self, feat_t, feat_x, ftx):
        feat_t_from_featx = bwarp(feat_x, ftx)
        return self.conv_res_feat(torch.cat((feat_t, feat_t_from_featx, ftx), dim=1))

    def _get_ref_coords(self, feat_t, ftx, movement_feat):
        b, c, fh, fw = feat_t.shape
        ftx_res_offset = 2.0 * torch.tanh(self.conv_res_offset(movement_feat))
        ftx_res_offset = ftx_res_offset.unsqueeze(1).view(b, self.n_groups * self.n_samples, 2, fh, fw)
        ftx_ref = (ftx_res_offset + ftx.unsqueeze(1))
        return ftx_ref

    def _get_ref_feats(self, feat, flow):
        b, c, fh, fw = feat.shape
        feat = feat.view(b * self.n_groups, self.n_group_c, fh, fw)
        xx = torch.linspace(-1.0, 1.0, fw).view(1, 1, 1, fw).expand(b, -1, fh, -1)
        yy = torch.linspace(-1.0, 1.0, fh).view(1, 1, fh, 1).expand(b, -1, -1, fw)
        grid = torch.cat([xx, yy], 1).to(feat).unsqueeze(1)
        flow_ = torch.cat([flow[:, :, 0:1, :, :] / (fw - 1.0) / 2.0,
                           flow[:, :, 1:2, :, :] / ((fh - 1.0) / 2.0)], dim=2)      # b, nG * nS, 2, fh, fw
        grid_ = (grid + flow_).view(b * self.n_groups, self.n_samples, 2, fh * fw).permute(0, 1, 3, 2)      # b * nG, nS, fh * fw, 2

        samples = F.grid_sample(input=feat, grid=grid_, mode='bilinear', align_corners=True)
        samples = samples.view(b, c, self.n_samples, -1)
        return samples  # b, c, nS, fh * fw

    def forward(self, feat_t, feat0, feat1, ft0, ft1):
        feat_t0_movement = self._get_movement_feats(feat_t, feat0, ft0)
        feat0_ref_coords = self._get_ref_coords(feat_t, ft0, feat_t0_movement)     # B * nG, nS, 2, H, W
        feat0_samples_kv = self._get_ref_feats(feat0, feat0_ref_coords)
        feat_t_attend_feat0 = self.attn(feat_t, feat0_samples_kv)
        feat_t_attend_feat0 = feat_t_attend_feat0 + self.mlp(feat_t_attend_feat0)

        feat_t1_movement = self._get_movement_feats(feat_t, feat1, ft1)
        feat1_ref_coords = self._get_ref_coords(feat_t, ft1, feat_t1_movement)
        feat1_samples_kv = self._get_ref_feats(feat1, feat1_ref_coords)
        feat_t_attend_feat1 = self.attn(feat_t, feat1_samples_kv)
        feat_t_attend_feat1 = feat_t_attend_feat1 + self.mlp(feat_t_attend_feat1)

        out = self.blendblock(torch.cat((feat_t_attend_feat0, feat_t_attend_feat1), dim=1))
        if self.pred_res_flow:
            res_ft0 = self.conv_res_flow(feat_t0_movement)
            up_ft0 = res_ft0 + 2.0 * resize(ft0, scale_factor=2.0)
            res_ft1 = self.conv_res_flow(feat_t1_movement)
            up_ft1 = res_ft1 + 2.0 * resize(ft1, scale_factor=2.0)
            return out, up_ft0, up_ft1
        return out


class DATv1(Basemodel):
    """
        t 시점의 query 생성 --> t 시점의 feature 와 유사하게
        Cross-frame attention으로 t 시점의 frame 복원
    """
    def __init__(self, args):
        super(DATv1, self).__init__(args)
        self.nf = args.nf

        self.cnn_encoder = ResEncoder(args.nf, args.enc_res_blocks)

        self.dcn_feat_t_builder = DCNInterFeatBuilder(args.nf)
        self.query_builder3 = nn.ConvTranspose2d(args.nf + 4, args.nf + 4, 4, 2, 1)
        self.dat_scale3 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=9,
                                                             n_groups=8, n_heads=8, mlp_ratio=args.mlp_ratio)

        self.query_builder2 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.dat_scale2 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=9,
                                                             n_groups=4, n_heads=4, mlp_ratio=args.mlp_ratio)

        self.query_builder1 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.dat_scale1 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=9,
                                                             n_groups=4, n_heads=4, mlp_ratio=args.mlp_ratio,
                                                             pred_res_flow=False)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0] / 255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        pseudo_gt_fwd_flow_viz = flow_tensor_to_np(inp_dict['f01'][0]) / 255.
        pseudo_gt_bwd_flow_viz = flow_tensor_to_np(inp_dict['f10'][0]) / 255.
        fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
        viz_flow = torch.cat((torch.from_numpy(pseudo_gt_fwd_flow_viz).cuda(), torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), torch.from_numpy(pseudo_gt_bwd_flow_viz).cuda()),
                             dim=-1)
        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
        }

    def forward(self, inp_dict):
        x0, x1, t, mean_ = self.normalize_w_rgb_mean(inp_dict['x0'], inp_dict['x1'], inp_dict['t'])
        feat0_1, feat0_2, feat0_3, feat0_4 = self.cnn_encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.cnn_encoder(x1)

        pred_feat_t_4, pred_ft0_4, pred_ft1_4 = self.dcn_feat_t_builder(feat0_4, feat1_4)
        pred_scale_3 = self.query_builder3(torch.cat((pred_feat_t_4, pred_ft0_4, pred_ft1_4), dim=1))
        pred_feat_t_3 = pred_scale_3[:, :self.nf, :, :]
        pred_ft0_3, pred_ft1_3 = pred_scale_3[:, self.nf:self.nf+2, :, :],  pred_scale_3[:, self.nf+2:self.nf+4, :, :]

        attended_feat_t_3, pred_ft0_2, pred_ft1_2 = self.dat_scale3(pred_feat_t_3, feat0_3, feat1_3, pred_ft0_3, pred_ft1_3)

        query_feat_t_2 = self.query_builder2(attended_feat_t_3)
        attended_feat_t_2, pred_ft0_1, pred_ft1_1 = self.dat_scale2(query_feat_t_2, feat0_2, feat1_2, pred_ft0_2, pred_ft1_2)

        query_feat_t_1 = self.query_builder1(attended_feat_t_2)
        attended_feat_t_1 = self.dat_scale1(query_feat_t_1, feat0_1, feat1_1, pred_ft0_1, pred_ft1_1)

        imgt_pred = self.generate_rgb_frame(attended_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        ft0, ft1 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        xt_ = xt - mean_
        _, _, feat_t_3, feat_t_4 = self.cnn_encoder(xt_)

        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)
        geo_loss = 0.01 * (self.gc_loss(pred_feat_t_3, feat_t_3) + self.gc_loss(pred_feat_t_4, feat_t_4))

        pred_ft0, pred_ft1 = resize(pred_ft0_1, 2) * 2., resize(pred_ft1_1, 2) * 2.
        robust_weight0 = get_robust_weight(pred_ft0, ft0, beta=0.3)
        robust_weight1 = get_robust_weight(pred_ft1, ft1, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(4.0 * resize(pred_ft0_2, 4.0) - ft0, weight=robust_weight0) + \
                               self.rb_loss(4.0 * resize(pred_ft1_2, 4.0) - ft1, weight=robust_weight1) + \
                               self.rb_loss(8.0 * resize(pred_ft0_3, 8.0) - ft0, weight=robust_weight0) + \
                               self.rb_loss(8.0 * resize(pred_ft1_3, 8.0) - ft1, weight=robust_weight1) + \
                               self.rb_loss(16.0 * resize(pred_ft0_4, 16.0) - ft0, weight=robust_weight0) + \
                               self.rb_loss(16.0 * resize(pred_ft1_4, 16.0) - ft1, weight=robust_weight1))
        total_loss = l1_loss + census_loss + geo_loss + distill_loss
        return {
            'frame_preds': [imgt_pred],
            'f01': pred_ft0,
            'f10': pred_ft1,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'geometry_loss': geo_loss.item(),
            'flow_loss': distill_loss.item(),
        }


class DATv1Poc(Basemodel):
    """
        t 시점의 query 생성 --> t 시점의 feature 와 유사하게
        Cross-frame attention으로 t 시점의 frame 복원
    """
    def __init__(self, args):
        super(DATv1Poc, self).__init__(args)
        self.nf = args.nf

        self.cnn_encoder = ResEncoder(args.nf, args.enc_res_blocks)

        self.dcn_feat_t_builder = DCNInterFeatBuilder(args.nf)
        self.query_builder3 = nn.ConvTranspose2d(args.nf + 4, args.nf + 4, 4, 2, 1)
        self.dat_scale3 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=9,
                                                             n_groups=8, n_heads=8, mlp_ratio=args.mlp_ratio,
                                                             pred_res_flow=False)

        self.query_builder2 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.dat_scale2 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=9,
                                                             n_groups=4, n_heads=4, mlp_ratio=args.mlp_ratio,
                                                             pred_res_flow=False)

        self.query_builder1 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.dat_scale1 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=9,
                                                             n_groups=4, n_heads=4, mlp_ratio=args.mlp_ratio,
                                                             pred_res_flow=False)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0] / 255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        # pseudo_gt_fwd_flow_viz = flow_tensor_to_np(inp_dict['f01'][0]) / 255.
        # pseudo_gt_bwd_flow_viz = flow_tensor_to_np(inp_dict['f10'][0]) / 255.
        # fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
        # bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
        # viz_flow = torch.cat((torch.from_numpy(pseudo_gt_fwd_flow_viz).cuda(), torch.from_numpy(fwd_flow_viz).cuda(),
        #                       torch.from_numpy(bwd_flow_viz).cuda(), torch.from_numpy(pseudo_gt_bwd_flow_viz).cuda()),
        #                      dim=-1)
        return {
            # 'flow': viz_flow,
            'pred': pred_concat[0],
        }

    def forward(self, inp_dict):
        x0, x1, t, mean_ = self.normalize_w_rgb_mean(inp_dict['x0'], inp_dict['x1'], inp_dict['t'])
        feat0_1, feat0_2, feat0_3, feat0_4 = self.cnn_encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.cnn_encoder(x1)

        ft0, ft1 = inp_dict['f01'], inp_dict['f10']
        pred_ft0_4, pred_ft1_4 = resize(ft0, 1 / 16) / 16., resize(ft1, 1 / 16) / 16.
        pred_ft0_3, pred_ft1_3 = resize(ft0, 1 / 8) / 8., resize(ft1, 1 / 8) / 8.
        pred_ft0_2, pred_ft1_2 = resize(ft0, 1 / 4) / 4., resize(ft1, 1 / 4) / 4.
        pred_ft0_1, pred_ft1_1 = resize(ft0, 1 / 2) / 2., resize(ft1, 1 / 2) / 2.

        pred_feat_t_4, _, _ = self.dcn_feat_t_builder(feat0_4, feat1_4)
        pred_scale_3 = self.query_builder3(torch.cat((pred_feat_t_4, pred_ft0_4, pred_ft1_4), dim=1))
        pred_feat_t_3 = pred_scale_3[:, :self.nf, :, :]
        # pred_ft0_3, pred_ft1_3 = pred_scale_3[:, self.nf:self.nf+2, :, :],  pred_scale_3[:, self.nf+2:self.nf+4, :, :]

        attended_feat_t_3 = self.dat_scale3(pred_feat_t_3, feat0_3, feat1_3, pred_ft0_3, pred_ft1_3)

        query_feat_t_2 = self.query_builder2(attended_feat_t_3)
        attended_feat_t_2 = self.dat_scale2(query_feat_t_2, feat0_2, feat1_2, pred_ft0_2, pred_ft1_2)

        query_feat_t_1 = self.query_builder1(attended_feat_t_2)
        attended_feat_t_1 = self.dat_scale1(query_feat_t_1, feat0_1, feat1_1, pred_ft0_1, pred_ft1_1)

        imgt_pred = self.generate_rgb_frame(attended_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        ft0, ft1 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        xt_ = xt - mean_
        _, _, feat_t_3, feat_t_4 = self.cnn_encoder(xt_)

        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)
        geo_loss = 0.01 * (self.gc_loss(pred_feat_t_3, feat_t_3) + self.gc_loss(pred_feat_t_4, feat_t_4))

        # pred_ft0, pred_ft1 = resize(pred_ft0_1, 2) * 2., resize(pred_ft1_1, 2) * 2.
        # robust_weight0 = get_robust_weight(pred_ft0, ft0, beta=0.3)
        # robust_weight1 = get_robust_weight(pred_ft1, ft1, beta=0.3)
        # distill_loss = 0.01 * (self.rb_loss(4.0 * resize(pred_ft0_2, 4.0) - ft0, weight=robust_weight0) + \
        #                        self.rb_loss(4.0 * resize(pred_ft1_2, 4.0) - ft1, weight=robust_weight1) + \
        #                        self.rb_loss(8.0 * resize(pred_ft0_3, 8.0) - ft0, weight=robust_weight0) + \
        #                        self.rb_loss(8.0 * resize(pred_ft1_3, 8.0) - ft1, weight=robust_weight1) + \
        #                        self.rb_loss(16.0 * resize(pred_ft0_4, 16.0) - ft0, weight=robust_weight0) + \
        #                        self.rb_loss(16.0 * resize(pred_ft1_4, 16.0) - ft1, weight=robust_weight1))
        total_loss = l1_loss + census_loss + geo_loss  # + distill_loss
        return {
            'frame_preds': [imgt_pred],
            # 'f01': pred_ft0,
            # 'f10': pred_ft1,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'geometry_loss': geo_loss.item(),
            # 'flow_loss': distill_loss.item(),
        }

