import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.warp import bwarp
from modules.corr import coords_grid, Mlp


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, act_at_last=True, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act_at_last = act_at_last
        if not act_at_last:
            self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :])
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :])
        out = x + self.conv5(out)
        if not self.act_at_last:
            out = self.prelu(out)
        return out


class SmallResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(SmallResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :])
        out = x + self.conv3(out)
        return out


class DeformAttnwMotion(nn.Module):
    def __init__(self, in_c, out_c, n_samples, n_heads):
        super(DeformAttnwMotion, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.q_proj = nn.Linear(in_c, out_c)
        self.k_proj = nn.Linear(in_c, out_c)
        self.v_proj = nn.Linear(in_c, out_c)

        self.n_samples = n_samples

        self.n_head = n_heads
        self.n_head_c = out_c // n_heads

        self.scale = self.n_head_c ** -0.5

    def forward(self, q, kv):
        """
            q: b, c, fh, fw
            kv0: b, c, nS, fh * fw
        """
        b, c, fh, fw = q.shape
        q = rearrange(q, 'b c fh fw -> b (fh fw) c')
        kv = rearrange(kv, 'b c nS (fh fw) -> b (nS fh fw) c', nS=self.n_samples, fh=fh, fw=fw)
        q = rearrange(self.q_proj(q), 'b f (nh nhc) -> (b nh) nhc f', nh=self.n_head, nhc=self.n_head_c)
        k = rearrange(self.k_proj(kv), 'b (nS fh fw) (nh nhc) -> (b nh) nhc nS (fh fw)',
                      nh=self.n_head, nhc=self.n_head_c, fh=fh, fw=fw, nS=self.n_samples)
        v = rearrange(self.v_proj(kv), 'b (nS fh fw) (nh nhc) -> (b nh) nhc nS (fh fw)',
                      nh=self.n_head, nhc=self.n_head_c, fh=fh, fw=fw, nS=self.n_samples)

        attn = torch.einsum('b c d, b c s d -> b s d', q, k).mul(self.scale)
        score = F.softmax(attn, dim=1)          # B * nH, nS, fh * fw
        out = torch.einsum('b s d, b c s d -> b c d', score, v).contiguous().view(b, self.out_c, fh, fw)

        kv0_attn, kv1_attn = torch.chunk(attn, 2, dim=1)

        # q = self.q_proj(q).view(b * self.n_head, self.n_head_c, fh * fw)
        # k = self.k_proj(kv).view(b * self.n_head, self.n_head_c, self.n_samples, fh * fw)
        # v = self.v_proj(kv).view(b * self.n_head, self.n_head_c, self.n_samples, fh * fw)

        # attn = torch.einsum('b c d, b c s d -> b s d', q, k)
        # attn = attn.mul(self.scale)             # B * nH, nS, fh * fw
        # score = F.softmax(attn, dim=1)          # B * nH, nS, fh * fw
        # out = torch.einsum('b s d, b c s d -> b c d', score, v).contiguous().view(b, self.out_c, fh, fw)
        # kv0_attn, kv1_attn = torch.chunk(attn, 2, dim=1)
        return out, F.softmax(kv0_attn, dim=1), F.softmax(kv1_attn, dim=1)


class DATwithMotionEstimation(nn.Module):
    def __init__(self, in_c, out_c, n_samples=9, n_groups=12, n_heads=12, mlp_ratio=2.0):
        super(DATwithMotionEstimation, self).__init__()
        assert n_groups == n_heads, "For now different number of n_heads and n_groups are not supported"
        self.in_c = in_c
        self.out_c = out_c
        self.n_groups = n_groups
        self.n_group_c = in_c // n_groups
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.norm1 = nn.LayerNorm(self.in_c)
        self.norm2 = nn.LayerNorm(self.out_c)
        self.mlp = Mlp(self.out_c, int(self.out_c * mlp_ratio))
        self.coord_proj = nn.Linear(2, self.out_c)
        self.motion_proj = nn.Linear(self.out_c, self.out_c)
        self.conv_res_feat = nn.Sequential(
            nn.Conv2d(self.in_c * 2 + 2, self.in_c, 3, 1, 1),
            nn.PReLU(self.in_c),
            SmallResBlock(self.in_c, self.in_c // 2)
        )
        # Calculate residual offset
        self.conv_res_offset = nn.Conv2d(self.in_c, n_groups * n_samples * 2, 3, 1, 1)
        self.conv_res_offset.apply(self._init_zero)

        self.attn = DeformAttnwMotion(in_c, out_c, n_samples * 2, n_heads)

    @staticmethod
    def _init_zero(m):
        if type(m) == nn.Conv2d:
            m.weight.data.zero_()
            m.bias.data.zero_()

    def _get_movement_feats(self, feat_t, feat_x, ftx):
        feat_t_from_featx = bwarp(feat_x, ftx)
        return self.conv_res_feat(torch.cat((feat_t, feat_t_from_featx, ftx), dim=1))

    def _get_ref_flow(self, feat_t, ftx, movement_feat):
        b, c, fh, fw = feat_t.shape
        ftx_res_offset = 2.0 * torch.tanh(self.conv_res_offset(movement_feat))
        ftx_res_offset = ftx_res_offset.unsqueeze(1).view(b, self.n_groups * self.n_samples, 2, fh, fw)
        ftx_ref = (ftx_res_offset + ftx.unsqueeze(1))
        return ftx_ref

    def _get_ref_feats(self, feat, flow):
        b, c, fh, fw = feat.shape
        feat = self.norm1(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        feat = feat.view(b * self.n_groups, self.n_group_c, fh, fw)
        xx = torch.linspace(-1.0, 1.0, fw).view(1, 1, 1, fw).expand(b, -1, fh, -1)
        yy = torch.linspace(-1.0, 1.0, fh).view(1, 1, fh, 1).expand(b, -1, -1, fw)
        grid = torch.cat([xx, yy], 1).to(feat).unsqueeze(1)
        flow_ = torch.cat([flow[:, :, 0:1, :, :] / (fw - 1.0) / 2.0,
                           flow[:, :, 1:2, :, :] / ((fh - 1.0) / 2.0)], dim=2)  # b, nG * nS, 2, fh, fw
        grid_ = (grid + flow_).view(b * self.n_groups, self.n_samples, 2, fh * fw).permute(0, 1, 3, 2)  # b * nG, nS, fh * fw, 2
        samples = F.grid_sample(input=feat, grid=grid_, mode='bilinear', align_corners=True)
        samples = samples.view(b, c, self.n_samples, -1)
        # [b, c, nS, fh * fw] [b, nG * nS, fh * fw, 2]
        return samples, grid_.view(b, self.n_groups * self.n_samples, fh * fw, 2)

    def forward(self, feat_t, feat0, feat1, ft0, ft1):
        b, c, fh, fw = feat_t.shape
        feat_t0_movement = self._get_movement_feats(feat_t, feat0, ft0)
        feat0_ref_flow = self._get_ref_flow(feat_t, ft0, feat_t0_movement)                          # B * nG, nS, 2, H, W
        feat0_samples_kv, feat0_sample_coords = self._get_ref_feats(feat0, feat0_ref_flow)         # b, c, nS, fh * fw

        feat_t1_movement = self._get_movement_feats(feat_t, feat1, ft1)
        feat1_ref_flow = self._get_ref_flow(feat_t, ft1, feat_t1_movement)
        feat1_samples_kv, feat1_sample_coords = self._get_ref_feats(feat1, feat1_ref_flow)

        feat_t = self.norm1(feat_t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_t_attend, feat0_sample_score, feat1_sample_score = self.attn(feat_t,
                                                                          torch.cat((feat0_samples_kv,
                                                                                     feat1_samples_kv), dim=2))
        feat_t = feat_t + feat_t_attend
        feat_t = rearrange(feat_t, 'b c fh fw -> b (fh fw) c')
        out = feat_t + self.mlp(self.norm2(feat_t), fh, fw)
        out = rearrange(out, 'b (fh fw) c -> b c fh fw', fh=fh, fw=fw)

        base_coord = coords_grid(b, fh, fw, feat_t.device)
        base_coord = rearrange(base_coord, 'b c h w -> b (h w) c')
        base_coord = self.coord_proj(base_coord)                            # b * nh, fh * fw, nhc
        feat0_sample_coord_proj = self.coord_proj(feat0_sample_coords)      # b, nG * nS, fh*fw, d
        feat1_sample_coord_proj = self.coord_proj(feat1_sample_coords)
        feat0_sample_score = feat0_sample_score.view(b, self.n_heads * self.n_samples, fh * fw)
        feat1_sample_score = feat1_sample_score.view(b, self.n_heads * self.n_samples, fh * fw)
        feat0_moved = torch.einsum('b s f, b s f c -> b f c', feat0_sample_score, feat0_sample_coord_proj)  # b, fh * fw, d
        feat1_moved = torch.einsum('b s f, b s f c -> b f c', feat1_sample_score, feat1_sample_coord_proj)  # b, fh * fw, d
        motion_t0 = self.motion_proj(feat0_moved - base_coord)
        motion_t1 = self.motion_proj(feat1_moved - base_coord)
        return out, motion_t0, motion_t1
