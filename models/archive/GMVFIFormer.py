import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules.losses as losses
from .BaseModel import GMBase
from modules.bwarp import warp
from modules.fwarp import SoftSplat
from modules.vfiformer import TFModel
from modules.backbone import CNNEncoder
from modules.attention import SelfAttnPropagation
from modules.transformer import FeatureTransformer
from modules.swin_utils import feature_add_position
from modules.matching import global_correlation_softmax
from modules.geometry import forward_backward_consistency_check, compute_out_of_boundary_mask


class GMVFIFormer(GMBase):
    def __init__(self, args):
        super(GMVFIFormer, self).__init__(args)
        self.l1 = nn.L1Loss()
        self.fuse_block = nn.Sequential(nn.Conv2d(12, 48, 3, 1, 1),
                                        nn.PReLU(48),
                                        nn.Conv2d(48, 48, 3, 1, 1),
                                        nn.PReLU(48))
        self.vfi_former = TFModel(img_size=(args.crop_h, args.crop_w), in_chans=48, out_chans=4, fuse_c=24,
                                  window_size=8, img_range=1.,
                                  depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                  embed_dim=160, num_heads=[[2, 2], [2, 2], [2, 2], [2, 2]], mlp_ratio=2,
                                  resi_connection='1conv',
                                  use_crossattn=[[[False, False, False, False], [True, True, True, True]], \
                                                 [[False, False, False, False], [True, True, True, True]], \
                                                 [[False, False, False, False], [True, True, True, True]], \
                                                 [[False, False, False, False], [False, False, False, False]]])

    def get_params_with_lr(self):
        return [
            {'params': self.backbone.parameters(), 'lr': self.args.small_lr},
            {'params': self.transformer.parameters(), 'lr': self.args.small_lr},
            {'params': self.upsampler.parameters(), 'lr': self.args.small_lr},
            {'params': self.feature_flow_attn.parameters(), 'lr': self.args.small_lr},
            {'params': self.alpha},
            {'params': self.fuse_block.parameters()},
            {'params': self.vfi_former.parameters()},
        ]

    def generate_base_frame(self, feat0, feat1, x0, x1, t):
        b, _, fh, fw = feat0.shape

        corr_flow_preds, dual_probs = global_correlation_softmax(feat0, feat1, pred_bidir_flow=True)

        # Self attention for flow refine
        feats = torch.cat((feat0, feat1), dim=0)  # [2*B, C, H, W] for propagation
        flow_preds = self.feature_flow_attn(feats, corr_flow_preds.detach())
        f01, f10 = torch.chunk(flow_preds, 2, dim=0)

        # Make upsample mask
        _for_mask = torch.cat((torch.cat((f01, feat0), dim=1), torch.cat((f10, feat1), dim=1)), dim=0)
        mask = self.upsampler(_for_mask)

        # Get origin size flow
        flow_up = self.upsample_with_mask(flow_preds, mask, 8) * self.upsample_factor
        f01_up, f10_up = torch.chunk(flow_up, 2, dim=0)

        # Calculate importance metric
        with torch.no_grad():
            matching_certainty = torch.max(dual_probs.detach(), dim=-1)[0].view(2 * b, 1, fh, fw)
            small_z0, small_z1 = torch.chunk(matching_certainty, 2)
            certainty_up = self.upsample_with_mask(matching_certainty, mask, 8)
            x0_certainty, x1_certainty = torch.chunk(certainty_up, 2)

        small_z0, small_z1 = small_z0 * self.alpha, small_z1 * self.alpha
        z0, z1 = x0_certainty * self.alpha, x1_certainty * self.alpha

        small_ft0 = -self.fwarper(f01, f01 * t, small_z0) * t
        # small_ft0_mask = compute_out_of_boundary_mask(small_ft0).unsqueeze(1)
        # small_ft0 = small_ft0 * small_ft0_mask

        small_ft1 = -self.fwarper(f10, f10 * (1 - t), small_z1) * (1 - t)
        # small_ft1_mask = compute_out_of_boundary_mask(small_ft1).unsqueeze(1)
        # small_ft1 = small_ft1 * small_ft1_mask

        # Warp frame
        if self.w_mode == 'f':
            xt_warp_x0, xt_warp_x1 = self.fwarp_frame(x0, x1, f01_up, f10_up, t, z0, z1)
        else:
            xt_warp_x0, xt_warp_x1, ft0_out_mask, ft1_out_mask = self.bwarp_frame(x0, x1, f01_up, f10_up, t, z0, z1)

        # Calculate blending mask
        zt_fwarp_z0 = self.fwarper(x0_certainty, f01_up * t, z0)
        zt_fwarp_z1 = self.fwarper(x1_certainty, f10_up * (1 - t), z1)

        fwd_occ, bwd_occ = forward_backward_consistency_check(f01, f10, alpha=0.01, beta=0.5)
        occ_up = self.upsample_with_mask(torch.concat((fwd_occ, bwd_occ), dim=0), mask, 8)
        fwd_occ_up, bwd_occ_up = torch.chunk(occ_up, 2, dim=0)

        occ_fwarp_fwd_occ = self.fwarper(fwd_occ_up, f01_up * t, z0)
        occ_fwarp_bwd_occ = self.fwarper(bwd_occ_up, f10_up * (1 - t), z1)

        x0_mask = zt_fwarp_z0 * (1 - zt_fwarp_z1) * (1 - occ_fwarp_bwd_occ)
        x1_mask = zt_fwarp_z1 * (1 - zt_fwarp_z0) * (1 - occ_fwarp_fwd_occ)
        if self.w_mode == 'b':
            x0_mask = x0_mask * ft0_out_mask
            x1_mask = x1_mask * ft1_out_mask

        x0_mask, x1_mask = x0_mask + 1e-6, x1_mask + 1e-6
        blended = (xt_warp_x0 * x0_mask + xt_warp_x1 * x1_mask) / (x0_mask + x1_mask)
        return blended, small_ft0, small_ft1, {
            'xt_warp_x0': xt_warp_x0,
            'xt_warp_x1': xt_warp_x1,
            'z0': z0,
            'z1': z1,
            'x0_mask': x0_mask,
            'x1_mask': x1_mask,
            'f01': f01_up,
            'f10': f10_up,
        }

    def calcul_losses(self, result_dict, inp_dict):
        final_pred = result_dict['frame_preds'][-1]
        gt_01 = inp_dict['xt'] / 255.

        l1_loss = (final_pred - gt_01).abs().mean()
        census_loss = self.census(final_pred, gt_01).mean()

        total = l1_loss + census_loss
        metric = {}
        metric.update({
            'total_loss': total.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'alpha': self.alpha[0].item(),
        })
        return total, metric

    def forward(self, inp_dict):
        results_dict = {}
        frame_preds = []
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']

        t = t.unsqueeze(-1).unsqueeze(-1)

        x0_normed = utils.normalize_imgnet(x0)
        x1_normed = utils.normalize_imgnet(x1)

        org_feat0, org_feat1 = self.extract_cnn_feature(x0=x0_normed, x1=x1_normed)
        feat0, feat1 = self.get_cross_attended_feature(org_feat0, org_feat1)

        base_frame, small_ft0, small_ft1, log_base = self.generate_base_frame(feat0, feat1, x0, x1, t)
        results_dict.update(log_base)
        frame_preds.append(base_frame)

        xt_warp_x0, xt_warp_x1 = log_base['xt_warp_x0'], log_base['xt_warp_x1']
        feat_t_from_0 = warp(org_feat0, small_ft0)
        feat_t_from_1 = warp(org_feat1, small_ft1)

        x = self.fuse_block(torch.cat([x0/255., x1/255., xt_warp_x0, xt_warp_x1], dim=1))

        refine_output = self.vfi_former(x, feat_t_from_0, feat_t_from_1)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = xt_warp_x0 * mask + xt_warp_x1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        frame_preds.append(pred)

        results_dict.update({
            'frame_preds': frame_preds,
            'alpha': self.alpha,
        })
        return results_dict


class GMVFIFormerV1(nn.Module):
    def __init__(self, args):
        super(GMVFIFormerV1, self).__init__()
        self.args = args

        # Loss
        self.l1 = nn.L1Loss()
        self.census = losses.Ternary()
        self.distill_lambda = args.distill_lambda
        self.robust_flow = losses.Charbonnier_Ada()

        self.fwarper = SoftSplat()
        self.w_mode = args.warp_mode
        self.alpha = nn.Parameter(torch.tensor([args.init_alpha]), requires_grad=False)

        self.backbone = CNNEncoder(output_dim=args.nf, num_output_scales=1)
        self.transformer = FeatureTransformer(num_layers=args.nlayer, d_model=args.nf,
                                              nhead=args.nhead, ffn_dim_expansion=args.ffc_expansion)
        self.feature_flow_attn = SelfAttnPropagation(in_channels=args.nf)

        self.load_gmflow = args.load_gmflow
        if args.load_gmflow:
            checkpoint = torch.load(args.load_gmflow)['model']
            for name, param in self.backbone.state_dict().items():
                param = checkpoint[f'backbone.{name}']
                self.backbone.state_dict()[name].copy_(param)
            for name, param in self.transformer.state_dict().items():
                param = checkpoint[f'transformer.{name}']
                self.transformer.state_dict()[name].copy_(param)
            for name, param in self.feature_flow_attn.state_dict().items():
                param = checkpoint[f'feature_flow_attn.{name}']
                self.feature_flow_attn.state_dict()[name].copy_(param)

        self.decoder = nn.Sequential(
            nn.Conv2d(args.nf * 4, args.nf * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(args.nf, args.nf * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(args.nf, args.nf * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(args.nf, args.nf // 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(args.nf // 2, 4, 3, 1, 1),
        )

    def extract_cnn_feature(self, x0, x1):
        concat = torch.cat((x0, x1), dim=0)
        features = self.backbone(concat)
        low_res_feat0, low_res_feat1 = torch.chunk(features[-1], 2, dim=0)
        return low_res_feat0, low_res_feat1

    def get_cross_attended_feature(self, feat0, feat1):
        feat0, feat1 = feature_add_position(feat0, feat1, 2, self.args.nf)
        feat0, feat1 = self.transformer(feat0, feat1, attn_num_splits=2)
        return feat0, feat1

    @staticmethod
    def upsample_flow(f01, mult=8):
        return F.interpolate(f01, scale_factor=mult, mode='bilinear', align_corners=True) * mult

    def get_reversed_flow(self, feat0, feat1, t):
        flow_preds = []

        b, _, fh, fw = feat0.shape

        corr_flow_preds, dual_probs = global_correlation_softmax(feat0, feat1, pred_bidir_flow=True)

        with torch.no_grad():
            matching_certainty = torch.max(dual_probs.detach(), dim=-1)[0].view(2 * b, 1, fh, fw) + 1e-6
            x0_certainty, x1_certainty = torch.chunk(matching_certainty, 2)
        z0 = x0_certainty * self.alpha
        z1 = x1_certainty * self.alpha

        flow_preds.append(self.upsample_flow(corr_flow_preds))

        feats = torch.cat((feat0, feat1), dim=0)
        flow_preds = self.feature_flow_attn(feats, corr_flow_preds.detach())
        f01, f10 = torch.chunk(flow_preds, 2, dim=0)

        ft0 = -self.fwarper(f01, f01 * t, z0) * t
        # ft0_mask = compute_out_of_boundary_mask(ft0).unsqueeze(1)
        # ft0 = ft0 * ft0_mask

        ft1 = -self.fwarper(f10, f10 * (1 - t), z1) * (1 - t)
        # ft1_mask = compute_out_of_boundary_mask(ft1).unsqueeze(1)
        # ft1 = ft1 * ft1_mask

        up_flow = self.upsample_flow(flow_preds)
        f01_up, f10_up = torch.chunk(up_flow, 2, dim=0)

        return ft0, ft1, f01_up, f10_up, z0, z1

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        t = t.unsqueeze(-1).unsqueeze(-1)
        x0_normed = utils.normalize_imgnet(x0)
        x1_normed = utils.normalize_imgnet(x1)

        org_feat0, org_feat1 = self.extract_cnn_feature(x0=x0_normed, x1=x1_normed)
        feat0, feat1 = self.get_cross_attended_feature(org_feat0, org_feat1)

        ft0, ft1, f01_up, f10_up, z0, z1 = self.get_reversed_flow(feat0, feat1, t)
        feat_t_from_0 = warp(org_feat0, ft0)
        feat_t_from_1 = warp(org_feat1, ft1)

        decoder_inp = torch.cat((feat0.detach(), feat_t_from_0, feat_t_from_1, feat1.detach()), dim=1)
        decoded = self.decoder(decoder_inp)
        res = torch.tanh(decoded[:, :3])
        mask = torch.sigmoid(decoded[:, 3:4])

        # ft0_up, ft1_up = self.upsample_flow(ft0), self.upsample_flow(ft1)
        z0_up = F.interpolate(z0, scale_factor=8, mode='bilinear', align_corners=True)
        z1_up = F.interpolate(z1, scale_factor=8, mode='bilinear', align_corners=True)
        xt_warp_x0 = self.fwarper(x0 / 255., f01_up * t, z0_up)
        xt_warp_x1 = self.fwarper(x1 / 255., f10_up * (1 - t), z1_up)

        merged_img = xt_warp_x0 * mask + xt_warp_x1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        return {
            'frame_preds': [merged_img, pred],
            'f01': f01_up,
            'f10': f10_up,
            'xt_warp_x0': xt_warp_x0,
            'xt_warp_x1': xt_warp_x1,
            'x0_mask': mask,
            'x1_mask': 1-mask,
        }

    def calcul_losses(self, result_dict, inp_dict):
        f01, f10 = inp_dict['f01'], inp_dict['f10']
        f01_pred, f10_pred = result_dict['f01'], result_dict['f10']
        flow_loss = self.robust_flow(f01_pred, f01) + self.robust_flow(f10_pred, f10)

        gt_01 = inp_dict['xt'] / 255.
        final_pred = result_dict['frame_preds'][-1]
        l1_loss = (final_pred - gt_01).abs().mean()
        census_loss = self.census(final_pred, gt_01).mean()

        total = l1_loss + census_loss + (flow_loss * self.distill_lambda)
        return total, {
            'total_loss': total.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'flow_loss': flow_loss.item(),
            'alpha': self.alpha[0].item(),
        }

    def get_params_with_lr(self):
        flow_lr = self.args.small_lr if self.load_gmflow else self.args.lr
        params = [
            {'params': self.decoder.parameters(), 'lr': self.args.lr},
            {'params': self.alpha, 'lr': self.args.lr},
            {'params': self.backbone.parameters(), 'lr': flow_lr},
            {'params': self.transformer.parameters(), 'lr': flow_lr},
            {'params': self.feature_flow_attn.parameters(), 'lr': flow_lr},
        ]
        return params
