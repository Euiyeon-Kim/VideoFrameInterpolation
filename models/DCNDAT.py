import torch
import torch.nn as nn
import torchvision.ops
import torch.nn.functional as F

from modules.warp import bwarp
from modules.dcnv2 import DeformableConv2d
from modules.residual_encoder import make_layer
from modules.deformable_attn import DeformAttn
from modules.losses import Ternary, Geometry, Charbonnier_Ada, Charbonnier_L1, get_robust_weight
from models.base import Basemodel
from models.DAT import Mlp
from models.IFRNet import convrelu, ResBlock
from utils import resize


class ResEncoder(nn.Module):
    def __init__(self, nf, n_res_block):
        super(ResEncoder, self).__init__()
        layers = [
            nn.Conv2d(3, nf, 3, 1, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
        ]
        if n_res_block > 0:
            layers.extend(make_layer(nf=nf, n_layers=n_res_block))
        self.projection = nn.Sequential(*layers)
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
        feat1 = self.projection(x)
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        feat4 = self.fea_L4_conv(feat3)
        return feat1, feat2, feat3, feat4


class DeformableConv2dGivenOffset(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, groups=8):
        super(DeformableConv2dGivenOffset, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.group = groups
        self.regular_conv = nn.Conv2d(in_channels=in_c // groups, out_channels=out_c,
                                      kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x, offset, mask):
        b, _, fh, fw = x.shape
        offset = offset.permute(0, 2, 1, 3, 4).flip(1).view(b, -1, fh, fw)
        mask = torch.sigmoid(mask)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x


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
        self.dcn = DeformableConv2d(nc, nc)
        self.blendblock = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 3, 1, 1),
            nn.PReLU(nc),
            nn.Conv2d(nc, nc, 3, 1, 1),
        )

    def forward(self, feat0, feat1, t):
        _, c, fh, fw = feat0.shape
        concat_t = t.repeat(1, 1, fh, fw)
        f01_offset_feat = self.convblock(torch.cat((feat0, feat1, concat_t), 1))
        f10_offset_feat = self.convblock(torch.cat((feat1, feat0, 1 - concat_t), 1))
        feat_t_from_feat0, ft0_offset = self.dcn(feat0, f01_offset_feat)
        feat_t_from_feat1, ft1_offset = self.dcn(feat1, f10_offset_feat)
        out = self.blendblock(torch.cat((feat_t_from_feat0, feat_t_from_feat1), 1))
        return out, ft0_offset, ft1_offset


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
        self.conv_res_offset_mask = nn.Conv2d(in_c, n_groups * n_samples * 3, 3, 1, 1)
        self.conv_res_offset_mask.apply(self._init_zero)
        if pred_res_flow:
            self.conv_res_flow = nn.ConvTranspose2d(in_c, 2, 4, 2, 1, bias=True)

        self.query_enhancer_dcn = DeformableConv2dGivenOffset(in_c, out_c)
        self.query_blender = nn.Conv2d(in_c * 3, in_c, 1, 1, 0)
        self.attn = DeformAttn(in_c, out_c, n_samples * 2, n_heads)
        self.mlp = Mlp(in_features=out_c, hidden_features=out_c * mlp_ratio, out_features=out_c)

    @staticmethod
    def _init_zero(m):
        if type(m) == nn.Conv2d:
            m.weight.data.zero_()
            m.bias.data.zero_()

    def _get_movement_feats(self, feat_t, feat_x, ftx):
        feat_t_from_featx = bwarp(feat_x, ftx)
        return self.conv_res_feat(torch.cat((feat_t, feat_t_from_featx, ftx), dim=1))

    def _get_ref_coords_mask(self, feat_t, ftx, movement_feat):
        b, c, fh, fw = feat_t.shape
        out = self.conv_res_offset_mask(movement_feat)
        res_o1, res_o2, mask = torch.chunk(out, 3, dim=1)
        ftx_res_offset = 2. * torch.tanh(torch.cat((res_o1, res_o2), dim=1))
        ftx_res_offset = ftx_res_offset.unsqueeze(1).view(b, self.n_groups * self.n_samples, 2, fh, fw)
        ftx_ref = (ftx_res_offset + ftx.unsqueeze(1))
        return ftx_ref, mask

    def _get_ref_feats(self, feat, flow):
        b, c, fh, fw = feat.shape
        feat = feat.view(b * self.n_groups, self.n_group_c, fh, fw)
        xx = torch.linspace(-1.0, 1.0, fw).view(1, 1, 1, fw).expand(b, -1, fh, -1)
        yy = torch.linspace(-1.0, 1.0, fh).view(1, 1, fh, 1).expand(b, -1, -1, fw)
        grid = torch.cat([xx, yy], 1).to(feat).unsqueeze(1)
        flow_ = torch.cat([flow[:, :, 0:1, :, :] / (fw - 1.0) / 2.0,
                           flow[:, :, 1:2, :, :] / ((fh - 1.0) / 2.0)], dim=2)  # b, nG * nS, 2, fh, fw
        grid_ = (grid + flow_).view(b * self.n_groups, self.n_samples, 2, fh * fw).permute(0, 1, 3,
                                                                                           2)  # b * nG, nS, fh * fw, 2

        samples = F.grid_sample(input=feat, grid=grid_, mode='bilinear', align_corners=True)
        samples = samples.view(b, c, self.n_samples, -1)
        return samples  # b, c, nS, fh * fw

    def forward(self, feat_t, feat0, feat1, ft0, ft1):
        feat_t0_movement = self._get_movement_feats(feat_t, feat0, ft0)
        feat0_ref_coords, feat0_dcn_mask = self._get_ref_coords_mask(feat_t, ft0, feat_t0_movement)  # B * nG, nS, 2, H, W
        feat_t1_movement = self._get_movement_feats(feat_t, feat1, ft1)
        feat1_ref_coords, feat1_dcn_mask = self._get_ref_coords_mask(feat_t, ft1, feat_t1_movement)  # B * nG, nS, 2, H, W
        feat_t_from_dcn_feat0 = self.query_enhancer_dcn(feat0, feat0_ref_coords, feat0_dcn_mask)
        feat_t_from_dcn_feat1 = self.query_enhancer_dcn(feat1, feat1_ref_coords, feat1_dcn_mask)
        feat_t_query = self.query_blender(torch.cat((feat_t_from_dcn_feat0, feat_t, feat_t_from_dcn_feat1), dim=1))

        feat0_samples_kv = self._get_ref_feats(feat0, feat0_ref_coords)  # b, c, nS, fh * fw
        feat1_samples_kv = self._get_ref_feats(feat1, feat1_ref_coords)
        feat_t_attend = self.attn(feat_t_query, torch.cat((feat0_samples_kv, feat1_samples_kv), dim=2))
        out = feat_t_attend + self.mlp(feat_t_attend)

        if self.pred_res_flow:
            res_ft0 = self.conv_res_flow(feat_t0_movement)
            up_ft0 = res_ft0 + 2.0 * resize(ft0, scale_factor=2.0)
            res_ft1 = self.conv_res_flow(feat_t1_movement)
            up_ft1 = res_ft1 + 2.0 * resize(ft1, scale_factor=2.0)
            return out, up_ft0, up_ft1
        return out


class DCNDATv1(Basemodel):
    def __init__(self, args):
        super(DCNDATv1, self).__init__(args)
        self.nf = args.nf

        self.cnn_encoder = ResEncoder(args.nf, args.enc_res_blocks)
        self.dcn_feat_t_builder = DCNInterFeatBuilderwithT(args.nf)
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
        self.geo_lambda = args.geo_lambda
        self.distill_lambda = args.distill_lambda
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def forward(self, inp_dict):
        x0, x1, t, mean_ = self.normalize_w_rgb_mean(inp_dict['x0'], inp_dict['x1'], inp_dict['t'])
        feat0_1, feat0_2, feat0_3, feat0_4 = self.cnn_encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.cnn_encoder(x1)
        pred_feat_t_4, pred_ft0_4, pred_ft1_4 = self.dcn_feat_t_builder(feat0_4, feat1_4, t)
        pred_scale_3 = self.query_builder3(torch.cat((pred_feat_t_4, pred_ft0_4, pred_ft1_4), dim=1))
        pred_feat_t_3 = pred_scale_3[:, :self.nf, :, :]
        pred_ft0_3, pred_ft1_3 = pred_scale_3[:, self.nf:self.nf+2, :, :],  pred_scale_3[:, self.nf+2:self.nf+4, :, :]

        attended_feat_t_3, pred_ft0_2, pred_ft1_2 = self.dat_scale3(pred_feat_t_3, feat0_3, feat1_3, pred_ft0_3, pred_ft1_3)

        query_feat_t_2 = self.query_builder2(attended_feat_t_3)
        attended_feat_t_2, pred_ft0_1, pred_ft1_1 = self.dat_scale2(query_feat_t_2, feat0_2, feat1_2, pred_ft0_2,
                                                                    pred_ft1_2)

        query_feat_t_1 = self.query_builder1(attended_feat_t_2)
        attended_feat_t_1 = self.dat_scale1(query_feat_t_1, feat0_1, feat1_1, pred_ft0_1, pred_ft1_1)

        imgt_pred = self.generate_rgb_frame(attended_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Prepare images for logging
        pred_ft0, pred_ft1 = resize(pred_ft0_1, 2) * 2., resize(pred_ft1_1, 2) * 2.
        img_dict = {'frame_preds': [imgt_pred], 'f0x': pred_ft0, 'f1x': pred_ft1}

        # Get GT to calculate losses
        ft0, ft1 = inp_dict['f0x'], inp_dict['f1x']
        xt = inp_dict['xt'] / 255

        # Photometric losses
        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)
        total_loss = l1_loss + census_loss
        log_dict = {
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
        }

        # Geometric loss
        if self.geo_lambda is not None:
            xt_ = xt - mean_
            _, _, feat_t_3, feat_t_4 = self.cnn_encoder(xt_)
            geo_loss = self.geo_lambda * (self.gc_loss(pred_feat_t_3, feat_t_3) + self.gc_loss(pred_feat_t_4, feat_t_4))
            total_loss = total_loss + geo_loss
            log_dict.update({'geometry_loss': geo_loss.item()})

        # Flow distillation loss
        if self.distill_lambda is not None:
            robust_weight0 = get_robust_weight(pred_ft0, ft0, beta=0.3)
            robust_weight1 = get_robust_weight(pred_ft1, ft1, beta=0.3)
            distill_loss = self.distill_lambda * (
                        self.rb_loss(4.0 * resize(pred_ft0_2, 4.0) - ft0, weight=robust_weight0) + \
                        self.rb_loss(4.0 * resize(pred_ft1_2, 4.0) - ft1, weight=robust_weight1) + \
                        self.rb_loss(8.0 * resize(pred_ft0_3, 8.0) - ft0, weight=robust_weight0) + \
                        self.rb_loss(8.0 * resize(pred_ft1_3, 8.0) - ft1, weight=robust_weight1) + \
                        self.rb_loss(16.0 * resize(pred_ft0_4, 16.0) - ft0, weight=robust_weight0) + \
                        self.rb_loss(16.0 * resize(pred_ft1_4, 16.0) - ft1, weight=robust_weight1))
            total_loss = total_loss + distill_loss
            log_dict.update({'flow_loss': distill_loss.item()})

        log_dict.update({'total_loss': total_loss.item()})
        return img_dict, total_loss, log_dict

