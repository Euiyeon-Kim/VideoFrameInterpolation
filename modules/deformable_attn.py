import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.warp import bwarp
from models.BaseModel import BaseModel
from modules.basic_blocks import conv_prelu, HalfChannelConv5ResBlock, FeadForward


class Attn(nn.Module):
    def __init__(self, in_c, out_c, n_samples, n_heads):
        super(Attn, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.q_proj = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.k_proj = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.v_proj = nn.Conv2d(in_c, out_c, 1, 1, 0)

        self.n_samples = n_samples

        self.n_head = n_heads
        self.n_head_c = out_c // n_heads

        self.scale = self.n_head_c ** -0.5

    def forward(self, q, kv):
        """
            q: b, c, fh, fw
            kv: b, c, nS, fh * fw
        """
        b, c, fh, fw = q.shape
        q = self.q_proj(q).view(b * self.n_head, self.n_head_c, fh * fw)
        k = self.k_proj(kv).view(b * self.n_head, self.n_head_c, self.n_samples, fh * fw)
        v = self.v_proj(kv).view(b * self.n_head, self.n_head_c, self.n_samples, fh * fw)

        attn = torch.einsum('b c d, b c s d -> b s d', q, k)
        attn = attn.mul(self.scale)         # B, nS, fh * fw
        attn = F.softmax(attn, dim=1)       # B, nS, fh * fw
        out = torch.einsum('b s d, b c s d -> b c d', attn, v).contiguous().view(b, self.out_c, fh, fw)
        return out


class CrossDeformableAttentionBlockwFlow(nn.Module):
    def __init__(self, in_c, out_c, n_samples=9, n_groups=12, n_heads=12, mlp_ratio=2.0, offset_scale=2.0,
                 pred_res_flow=True):
        super(CrossDeformableAttentionBlockwFlow, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.n_groups = n_groups
        self.n_group_c = in_c // n_groups
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.offset_scale = offset_scale
        self.pred_res_flow = pred_res_flow

        self.movement_extractor = nn.Sequential(
            conv_prelu(self.in_c * 2 + 2, self.in_c * 2),
            conv_prelu(self.in_c * 2, self.in_c),
            HalfChannelConv5ResBlock(self.in_c, self.in_c // 2)
        )

        # Calculate residual offset
        self.conv_res_offset = nn.Conv2d(self.in_c, n_groups * n_samples * 2, 3, 1, 1)
        self.conv_res_offset.apply(self._init_zero)
        if pred_res_flow:
            self.conv_res_flow = nn.ConvTranspose2d(self.in_c, 2, 4, 2, 1, bias=True)

        self.attn = Attn(in_c, out_c, n_samples * 2, n_heads)
        self.mlp = FeadForward(in_dim=out_c, hidden_dim=out_c * mlp_ratio, out_dim=out_c)

    @staticmethod
    def _init_zero(m):
        if type(m) == nn.Conv2d:
            m.weight.data.zero_()
            m.bias.data.zero_()

    def _get_movement_feats(self, feat_t, feat_x, ftx):
        feat_t_from_featx = bwarp(feat_x, ftx)
        return self.movement_extractor(torch.cat((feat_t, feat_t_from_featx, ftx), dim=1))

    def _get_ref_coords(self, feat_t, ftx, movement_feat):
        b, c, fh, fw = feat_t.shape
        ftx_res_offset = self.offset_scale * torch.tanh(self.conv_res_offset(movement_feat))
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
                           flow[:, :, 1:2, :, :] / ((fh - 1.0) / 2.0)], dim=2)  # b, nG * nS, 2, fh, fw
        grid_ = (grid + flow_).view(b * self.n_groups, self.n_samples, 2, fh * fw).permute(0, 1, 3,
                                                                                           2)  # b * nG, nS, fh * fw, 2

        samples = F.grid_sample(input=feat, grid=grid_, mode='bilinear', align_corners=True)
        samples = samples.view(b, c, self.n_samples, -1)
        return samples  # b, c, nS, fh * fw

    def forward(self, feat_t, feat0, feat1, ft0, ft1):
        # Select features from frame 0
        feat_t0_movement = self._get_movement_feats(feat_t, feat0, ft0)
        feat0_ref_coords = self._get_ref_coords(feat_t, ft0, feat_t0_movement)  # B * nG, nS, 2, H, W
        feat0_samples_kv = self._get_ref_feats(feat0, feat0_ref_coords)  # b, c, nS, fh * fw

        # Select features from frame 1
        feat_t1_movement = self._get_movement_feats(feat_t, feat1, ft1)
        feat1_ref_coords = self._get_ref_coords(feat_t, ft1, feat_t1_movement)
        feat1_samples_kv = self._get_ref_feats(feat1, feat1_ref_coords)

        # Attention with selected features
        feat_t_attend = self.attn(feat_t, torch.cat((feat0_samples_kv, feat1_samples_kv), dim=2))
        out = feat_t_attend + self.mlp(feat_t_attend)

        if self.pred_res_flow:
            res_ft0 = self.conv_res_flow(feat_t0_movement)
            up_ft0 = res_ft0 + 2.0 * BaseModel.resize(ft0, scale_factor=2.0)
            res_ft1 = self.conv_res_flow(feat_t1_movement)
            up_ft1 = res_ft1 + 2.0 * BaseModel.resize(ft1, scale_factor=2.0)
            return out, up_ft0, up_ft1
        return out

