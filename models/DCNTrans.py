import torch
import torch.nn as nn

from modules.losses import *
from models.GMTrans import Decoder2, resize
from modules.residual_encoder import make_layer
from utils.flow_viz import flow_tensor_to_np
from modules.positional_encoding import PositionEmbeddingSine
from modules.dcnv2 import DeformableConv2d, DeformableConv2dwithFwarpv2


class DCNInterFeatBuilder(nn.Module):
    """
        Backward warping으로 query building
    """
    def __init__(self, nc):
        super(DCNInterFeatBuilder, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.PReLU(nc),
        )
        self.dcn0t = DeformableConv2d(nc, nc)
        self.dcn1t = DeformableConv2d(nc, nc)
        self.blendblock = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
        )

    def forward(self, feat0, feat1):
        f01_offset_feat = self.convblock(torch.cat((feat0, feat1), 1))
        f10_offset_feat = self.convblock(torch.cat((feat1, feat0), 1))
        feat_t_from_feat0, ft0_offset = self.dcn0t(feat0, f01_offset_feat)
        feat_t_from_feat1, ft1_offset = self.dcn1t(feat1, f10_offset_feat)
        out = self.blendblock(torch.cat((feat_t_from_feat0, feat_t_from_feat1), 1))
        return out, ft0_offset, ft1_offset


class DCNTransv1(nn.Module):
    def __init__(self, args):
        super(DCNTransv1, self).__init__()
        self.args = args

        # Feature extraction
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.feature_extraction = make_layer(nf=args.nf, n_layers=args.enc_res_blocks)
        self.fea_L2_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.fea_L3_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.dcn_feat_t_builder = DCNInterFeatBuilder(nc=args.nf)

        self.pos_enc = PositionEmbeddingSine(num_pos_feats=args.nf // 2)
        self.query_builder2 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.decoder2 = Decoder2(inp_dim=args.nf, depth=8, num_heads=8, window_size=4, mlp_ratio=args.mlp_ratio)

        self.query_builder1 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.decoder1 = Decoder2(inp_dim=args.nf, depth=8, num_heads=4, window_size=4, mlp_ratio=args.mlp_ratio)

        self.reconstruction = make_layer(nf=args.nf, n_layers=args.dec_res_blocks)
        self.upconv1 = nn.Conv2d(args.nf, args.nf * 4, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU(args.nf)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(args.nf, args.nf, 3, 1, 1)
        self.prelu2 = nn.PReLU(args.nf)
        self.conv_last = nn.Conv2d(args.nf, 3, 3, 1, 1)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.gc_loss = Geometry(3)
        self.rb_loss = Charbonnier_Ada()
        # self.offset_lambda = args.offset_lambda
        # self.offset_loss = OffsetFidelityLoss(threshold=args.offset_thresh)

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0]/255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        pseudo_gt_fwd_flow_viz = flow_tensor_to_np(inp_dict['f01'][0]) / 255.
        pseudo_gt_bwd_flow_viz = flow_tensor_to_np(inp_dict['f10'][0]) / 255.
        fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
        viz_flow = torch.cat((torch.from_numpy(pseudo_gt_fwd_flow_viz).cuda(), torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), torch.from_numpy(pseudo_gt_bwd_flow_viz).cuda()), dim=-1)
        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
        }

    def extract_feature(self, x):
        feat1 = self.feature_extraction(self.conv_first(x))
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        return feat1, feat2, feat3

    def generate_rgb_frame(self, feat, m):
        out = self.reconstruction(feat)
        out = self.prelu1(self.pixel_shuffle(self.upconv1(out)))
        out = self.prelu2(self.HRconv(out))
        out = self.conv_last(out)
        return torch.clamp(out + m, 0, 1)

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1 = x0 / 255., x1 / 255.
        mean_ = (torch.mean(x0, dim=(2, 3), keepdim=True) + torch.mean(x1, dim=(2, 3), keepdim=True)) / 2
        x0_, x1_ = x0 - mean_, x1 - mean_

        feat0_1, feat0_2, feat0_3 = self.extract_feature(x0_)
        feat1_1, feat1_2, feat1_3 = self.extract_feature(x1_)

        # Pred with DCN
        pred_feat_t_3, ft0_offset, ft1_offset = self.dcn_feat_t_builder(feat0_3, feat1_3)

        # Attention
        pred_feat_t_2 = self.query_builder2(pred_feat_t_3)
        feat_t_2_query = pred_feat_t_2 + self.pos_enc(pred_feat_t_2)
        attended_feat_t_2 = self.decoder2(feat_t_2_query, feat0_2, feat1_2)

        feat_t_1_query = self.query_builder1(attended_feat_t_2)
        attended_feat_t_1 = self.decoder1(feat_t_1_query, feat0_1, feat1_1)

        imgt_pred = self.generate_rgb_frame(attended_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        ft0, ft1 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        xt_ = xt - mean_
        _, ft_2, ft_3 = self.extract_feature(xt_)

        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)
        geo_loss = 0.01 * (self.gc_loss(pred_feat_t_3, ft_3) +
                           self.gc_loss(pred_feat_t_2, ft_2))

        pred_ft0_offset, pred_ft1_offset = resize(ft0_offset, 8) * 8., resize(ft1_offset, 8) * 8.
        robust_weight0 = get_robust_weight(pred_ft0_offset, ft0, beta=0.3)
        robust_weight1 = get_robust_weight(pred_ft1_offset, ft1, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(pred_ft0_offset - ft0, weight=robust_weight0) +
                               self.rb_loss(pred_ft1_offset - ft1, weight=robust_weight1))

        total_loss = l1_loss + census_loss + geo_loss + distill_loss

        return {
            'frame_preds': [imgt_pred],
            'f01': pred_ft0_offset,
            'f10': pred_ft1_offset,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'geometry_loss': geo_loss.item(),
            'flow_loss': distill_loss.item(),
        }


class DCNInterFeatBuilderv2(nn.Module):
    """
        Average forward warping으로 query building
    """
    def __init__(self, nc):
        super(DCNInterFeatBuilderv2, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.PReLU(nc),
        )
        self.dcn0t = DeformableConv2dwithFwarpv2(nc, nc)
        self.dcn1t = DeformableConv2dwithFwarpv2(nc, nc)
        self.blendblock = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
        )

    def forward(self, feat0, feat1, t):
        f01_offset_feat = self.convblock(torch.cat((feat0, feat1), 1))
        f10_offset_feat = self.convblock(torch.cat((feat1, feat0), 1))
        feat_t_from_feat0, f01 = self.dcn0t(feat0, t, f01_offset_feat)
        feat_t_from_feat1, f10 = self.dcn1t(feat1, 1 - t, f10_offset_feat)
        out = self.blendblock(torch.cat((feat_t_from_feat0, feat_t_from_feat1), 1))
        return out, f01, f10


class DCNTransv2(nn.Module):
    def __init__(self, args):
        super(DCNTransv2, self).__init__()
        self.args = args

        # Feature extraction
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.feature_extraction = make_layer(nf=args.nf, n_layers=args.enc_res_blocks)
        self.fea_L2_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.fea_L3_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )

        self.dcn_feat_t_builder = DCNInterFeatBuilderv2(nc=args.nf)
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=args.nf // 2)

        self.query_builder2 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.decoder2 = Decoder2(inp_dim=args.nf, depth=8, num_heads=8, window_size=4, mlp_ratio=args.mlp_ratio)

        self.query_builder1 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.decoder1 = Decoder2(inp_dim=args.nf, depth=8, num_heads=4, window_size=4, mlp_ratio=args.mlp_ratio)

        self.reconstruction = make_layer(nf=args.nf, n_layers=args.dec_res_blocks)
        self.upconv1 = nn.Conv2d(args.nf, args.nf * 4, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU(args.nf)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(args.nf, args.nf, 3, 1, 1)
        self.prelu2 = nn.PReLU(args.nf)
        self.conv_last = nn.Conv2d(args.nf, 3, 3, 1, 1)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.gc_loss = Geometry(3)
        self.rb_loss = Charbonnier_Ada()

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0]/255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        pseudo_gt_fwd_flow_viz = flow_tensor_to_np(inp_dict['f01'][0]) / 255.
        pseudo_gt_bwd_flow_viz = flow_tensor_to_np(inp_dict['f10'][0]) / 255.
        fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
        viz_flow = torch.cat((torch.from_numpy(pseudo_gt_fwd_flow_viz).cuda(), torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), torch.from_numpy(pseudo_gt_bwd_flow_viz).cuda()), dim=-1)
        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
        }

    def extract_feature(self, x):
        feat1 = self.feature_extraction(self.conv_first(x))
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        return feat1, feat2, feat3

    def generate_rgb_frame(self, feat, m):
        out = self.reconstruction(feat)
        out = self.prelu1(self.pixel_shuffle(self.upconv1(out)))
        out = self.prelu2(self.HRconv(out))
        out = self.conv_last(out)
        return torch.clamp(out + m, 0, 1)

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1 = x0 / 255., x1 / 255.
        mean_ = (torch.mean(x0, dim=(2, 3), keepdim=True) + torch.mean(x1, dim=(2, 3), keepdim=True)) / 2
        x0_, x1_ = x0 - mean_, x1 - mean_

        feat0_1, feat0_2, feat0_3 = self.extract_feature(x0_)
        feat1_1, feat1_2, feat1_3 = self.extract_feature(x1_)

        # Pred with DCN
        pred_feat_t_3, pred_ft0_3, pred_ft1_3 = self.dcn_feat_t_builder(feat0_3, feat1_3, t)
        pred_feat_t_2 = self.query_builder2(pred_feat_t_3)

        # Attention
        feat_t_2_query = self.pos_enc(pred_feat_t_2) + pred_feat_t_2
        attended_feat_t_2 = self.decoder2(feat_t_2_query, feat0_2, feat1_2)

        feat_t_1_query = self.query_builder1(attended_feat_t_2)
        attended_feat_t_1 = self.decoder1(feat_t_1_query, feat0_1, feat1_1)

        imgt_pred = self.generate_rgb_frame(attended_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        ft0, ft1 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        xt_ = xt - mean_
        _, feat_t_2, feat_t_3 = self.extract_feature(xt_)

        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)
        geo_loss = 0.01 * (self.gc_loss(pred_feat_t_3, feat_t_3) +
                           self.gc_loss(pred_feat_t_2, feat_t_2))

        pred_ft0, pred_ft1 = resize(pred_ft0_3, 8) * 8., resize(pred_ft1_3, 8) * 8.
        robust_weight0 = get_robust_weight(pred_ft0, ft0, beta=0.3)
        robust_weight1 = get_robust_weight(pred_ft1, ft1, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(pred_ft0 - ft0, weight=robust_weight0) +
                               self.rb_loss(pred_ft1 - ft1, weight=robust_weight1))

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
