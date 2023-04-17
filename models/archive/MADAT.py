"""
    Get motion by cross-frame attention
    Deformable attention to generate intermediate frame
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.refine import Unet
from modules.corr import InterFrameAttention
from modules.warp import fwarp_using_two_frames, bwarp
from modules.madat import DATwithMotionEstimation
from utils.flow_viz import flow_tensor_to_np


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.layer = nn.Linear(3, 64)

    def get_log_dict(self, inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0] / 255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        pseudo_gt_fwd_flow_viz = flow_tensor_to_np(inp_dict['f0x'][0]) / 255.
        pseudo_gt_bwd_flow_viz = flow_tensor_to_np(inp_dict['f1x'][0]) / 255.
        fwd_flow_viz = flow_tensor_to_np(results_dict['f0x'][0][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f1x'][0][0]) / 255.
        viz_flow = torch.cat((torch.from_numpy(pseudo_gt_fwd_flow_viz).cuda(), torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), torch.from_numpy(pseudo_gt_bwd_flow_viz).cuda()),
                             dim=-1)

        ft0_2 = flow_tensor_to_np(self.resize(results_dict['f0x'][1], 2.0)[0] * 2.0) / 255.
        ft1_2 = flow_tensor_to_np(self.resize(results_dict['f1x'][1], 2.0)[0] * 2.0) / 255.
        ft0_3 = flow_tensor_to_np(self.resize(results_dict['f0x'][2], 4.0)[0] * 4.0) / 255.
        ft1_3 = flow_tensor_to_np(self.resize(results_dict['f1x'][2], 4.0)[0] * 4.0) / 255.
        flow_progress = torch.cat((torch.from_numpy(ft0_3).cuda(), torch.from_numpy(ft0_2).cuda(),
                                   torch.from_numpy(ft1_2).cuda(), torch.from_numpy(ft1_3).cuda()),
                                  dim=-1)
        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
            'process': flow_progress
        }

    @staticmethod
    def normalize_w_rgb_mean(x0, x1, t):
        """
            return 0-centered x0, x1
        """
        t = t.unsqueeze(-1).unsqueeze(-1)

        x0, x1 = x0 / 255., x1 / 255.
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_
        return x0, x1, t, mean_

    @staticmethod
    def resize(x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor,
                             recompute_scale_factor=False, mode="bilinear", align_corners=True)

    def inference(self, x0, x1, t):
        return (x0 + x1) / 2

    def forward(self, x0, x1, t):
        return (x0 + x1) / t


class FeatPyramid(nn.Module):
    def __init__(self, channels=(16, 32, 64), depths=(3, 3, 3)):
        super(FeatPyramid, self).__init__()
        assert len(channels) == len(depths)
        blocks = []
        channels = [3] + channels
        for i in range(len(channels) - 1):
            cur_layers = [
                nn.Conv2d(channels[i], channels[i + 1], 3, 2, 1),
                nn.PReLU(channels[i + 1], init=0.1)
            ]
            for _ in range(depths[i] - 1):
                cur_layers = cur_layers + [
                    nn.Conv2d(channels[i + 1], channels[i + 1], 3, 1, 1),
                    nn.PReLU(channels[i + 1], init=0.1)
                ]
            blocks.append(nn.Sequential(*cur_layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = []
        for blk in self.blocks:
            x = blk(x)
            out.append(x)
        return out


class SelfAttnPropagation(nn.Module):
    def __init__(self, in_c):
        super(SelfAttnPropagation, self).__init__()
        self.q_proj = nn.Linear(in_c, in_c)
        self.k_proj = nn.Linear(in_c, in_c)
        self.v_proj = nn.Linear(in_c, in_c)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_t, flow0, flow1):
        b, c, h, w = feat_t.size()
        query = feat_t.view(b, c, h * w).permute(0, 2, 1)             # [B, H*W, C]
        flow0 = flow0.view(b, flow0.size(1), h * w).permute(0, 2, 1)     # [B, H*W, 2]
        flow1 = flow1.view(b, flow1.size(1), h * w).permute(0, 2, 1)     # [B, H*W, 2]

        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)    # [B, H*W, C]
        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        refined_flow0 = torch.matmul(prob, flow0)  # [B, H*W, 2]
        refined_flow1 = torch.matmul(prob, flow1)  # [B, H*W, 2]

        value = self.v_proj(query)
        refined_feat_t = torch.matmul(prob, value)

        refined_flow0 = refined_flow0.view(b, h, w, flow0.size(-1)).permute(0, 3, 1, 2)                 # [B, 2, H, W]
        refined_flow1 = refined_flow1.view(b, h, w, flow1.size(-1)).permute(0, 3, 1, 2)                 # [B, 2, H, W]
        refined_feat_t = refined_feat_t.view(b, h, w, refined_feat_t.size(-1)).permute(0, 3, 1, 2)    # [B, C, H, W]

        return refined_feat_t, refined_flow0, refined_flow1


class MADATv1(BaseModel):
    def __init__(self, args):
        super(MADATv1, self).__init__(args)
        self.encoder = FeatPyramid(args.channels, args.depths)
        self.corr = InterFrameAttention(dim=args.channels[-1], motion_dim=64, mlp_ratio=args.mlp_ratio)
        self.motions2bwarpflow = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 4, 3, 1, 1),
        )
        self.motion2flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 2, 3, 1, 1),
        )
        self.self_attention_t_3 = SelfAttnPropagation(args.channels[-1])

        self.query_builder2 = nn.ConvTranspose2d(args.channels[-1] * 3, args.channels[-2], 4, 2, 1)
        self.decoder2 = DATwithMotionEstimation(args.channels[-2], args.channels[-2],
                                                n_samples=9, n_groups=4, n_heads=4, mlp_ratio=args.mlp_ratio)
        self.self_attention_t_2 = SelfAttnPropagation(args.channels[-2])

        self.query_builder1 = nn.ConvTranspose2d(args.channels[-2] * 3, args.channels[-3], 4, 2, 1)
        self.decoder1 = DATwithMotionEstimation(args.channels[-3], args.channels[-3],
                                                n_samples=9, n_groups=8, n_heads=8, mlp_ratio=args.mlp_ratio)
        self.self_attention_t_1 = SelfAttnPropagation(args.channels[-3])
        self.rgb_builder = Unet(args.channels)

    def forward(self, inp_dict):
        x0, x1, t, mean_ = self.normalize_w_rgb_mean(inp_dict['x0'], inp_dict['x1'], inp_dict['t'])
        feat0_1, feat0_2, feat0_3 = self.encoder(x0)
        feat1_1, feat1_2, feat1_3 = self.encoder(x1)

        b, _, fh, fw = feat0_3.shape

        # InterFrame attention and get movement features
        # Motion may be aligned at 0 or 1 timestamp
        cross_attended_feat_s, motion = self.corr(feat0_3, feat1_3)
        cross_attended_feat0_3, cross_attended_feat1_3 = torch.chunk(cross_attended_feat_s, 2, dim=0)
        motion_01, motion_10 = torch.chunk(motion, 2, dim=0)
        motion_0t, motion_1t = motion_01 * t, motion_10 * (1 - t)
        f0t_3, f1t_3 = torch.chunk(self.motion2flow(torch.cat((motion_0t, motion_1t), dim=0)), 2, dim=0)

        # Generate feature t with average fwarp
        _ones = torch.ones(b, 1, fh, fw).to(cross_attended_feat0_3.device)
        feat_t_3, _ = fwarp_using_two_frames(cross_attended_feat0_3, f0t_3, t,
                                             cross_attended_feat1_3, f1t_3, 1-t, _ones, _ones)

        ft0_3, ft1_3 = torch.chunk(self.motions2bwarpflow(torch.cat((motion_0t, motion_1t, feat_t_3), dim=1)), 2, dim=1)
        self_attended_feat_t_3, ft0_3, ft1_3 = self.self_attention_t_3(feat_t_3, ft0_3, ft1_3)
        feat_t_3_from_feat0_3 = bwarp(feat0_3, ft0_3)
        feat_t_3_from_feat1_3 = bwarp(feat1_3, ft1_3)

        feat_t_2_query = self.query_builder2(torch.cat((feat_t_3_from_feat0_3, self_attended_feat_t_3, feat_t_3_from_feat1_3), dim=1))
        up_ft0_3 = self.resize(ft0_3, 2.0) * 2.0
        up_ft1_3 = self.resize(ft1_3, 2.0) * 2.0
        feat_t_2, motion_t0_2, motion_t1_2 = self.decoder2(feat_t_2_query, feat0_2, feat1_2, up_ft0_3, up_ft1_3)

        res_ft0_2, res_ft1_2 = torch.chunk(self.motion2flow(torch.cat((motion_t0_2, motion_t1_2), dim=0)), 2, dim=0)
        ft0_2 = up_ft0_3 + res_ft0_2
        ft1_2 = up_ft1_3 + res_ft1_2

        self_attended_feat_t_2, ft0_2, ft1_2 = self.self_attention_t_2(feat_t_2, ft0_2, ft1_2)
        feat_t_2_from_feat0_2 = bwarp(feat0_2, ft0_2)
        feat_t_2_from_feat1_2 = bwarp(feat1_2, ft1_2)

        feat_t_1_query = self.query_builder1(torch.cat((feat_t_2_from_feat0_2, self_attended_feat_t_2, feat_t_2_from_feat1_2), dim=1))
        up_ft0_2 = self.resize(ft0_2, 2.0) * 2.0
        up_ft1_2 = self.resize(ft1_2, 2.0) * 2.0
        feat_t_1, motion_t0_1, motion_t1_1 = self.decoder1(feat_t_1_query, feat0_1, feat1_1, up_ft0_2, up_ft1_2)

        res_ft0_1, res_ft1_1 = torch.chunk(self.motion2flow(torch.cat((motion_t0_1, motion_t1_1), dim=0)), 2, dim=0)
        ft0_1 = up_ft0_2 + res_ft0_1
        ft1_1 = up_ft1_2 + res_ft1_1

        self_attended_feat_t_1, ft0_1, ft1_1 = self.self_attention_t_1(feat_t_1, ft0_1, ft1_1)

        up_ft0_1 = self.resize(ft0_1, 2.0) * 2.0
        up_ft1_1 = self.resize(ft1_1, 2.0) * 2.0
        xt_from_x0 = bwarp(x0, up_ft0_1)
        xt_from_x1 = bwarp(x1, up_ft1_1)

        rgb = self.rgb_builder(xt_from_x0, xt_from_x1,
                               [self_attended_feat_t_1, self_attended_feat_t_2, self_attended_feat_t_3])
        imgt_pred = torch.clamp(rgb + mean_, 0, 1)

        if not self.training:
            return imgt_pred

        # Get GT for logging
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
        img_dict = {'frame_preds': [imgt_pred],
                    'f0x': [up_ft0_1, up_ft0_2, up_ft1_3],
                    'f1x': [up_ft1_1, up_ft1_2, up_ft1_3]}
        log_dict.update({'total_loss': total_loss.item()})
        return img_dict, total_loss, log_dict
