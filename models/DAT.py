import einops

import torch
import torch.nn as nn

from modules.warp import bwarp
from models.base import Basemodel
from modules.deformable_attn import DeformAttn
from modules.residual_encoder import make_layer
from models.DCNTrans import DCNInterFeatBuilder


class ResEncoder(nn.Module):
    def __init__(self, nf, n_res_block):
        super(ResEncoder, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(3, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
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


class CrossDeformableAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, attn_window=(3, 3), groups=12, n_heads=12, mlp_ratio=2.0):
        self.in_c = in_c
        self.out_c = out_c
        self.groups = groups
        self.n_heads = n_heads
        self.attn_window = attn_window

        # Calculate residual offset
        self.conv_res_offset = nn.Sequential(
            nn.Conv2d(self.in_c * 2 + 2, self.in_c * 2, 3, 1, 1),
            nn.PReLU(self.in_c * 2),
            nn.Conv2d(self.in_c * 2, self.in_c, 3, 1, 1),
            nn.PReLU(self.in_c),
            nn.Conv2d(self.in_c, groups * attn_window[0] * attn_window[1] * 2, 3, 1, 1),
        )
        self.conv_res_offset.apply(self.init_zero)

        self.attn = DeformAttn(in_c, out_c, attn_window, groups, n_heads, mlp_ratio)

    @staticmethod
    def init_zero(m):
        if type(m) == nn.Conv2d:
            m.weight.data.zero_()
            m.bias.data.zero_()

    def forward(self, feat_t, feat0, feat1, ft0, ft1):
        kv_to_q = bwarp(kv_feat_x, flow_tx)
        res_offset = self.conv_res_offset(torch.cat((q_feat_t, kv_to_q, flow_tx)))
        offset = res_offset + flow_tx.flip(1).repeat(1, res_offset.size(1) // 2, 1, 1)
        print(offset.shape)
        exit()

        exit()


class DATv1(Basemodel):
    """
        t 시점의 query 생성 --> t 시점의 feature 와 유사하게
        Cross-frame attention으로 t 시점의 frame 복원
    """
    def __init__(self, args):
        super(DATv1, self).__init__(args)
        self.nf = args.nf

        self.cnn_encoder = ResEncoder(args.nf, args.enc_res_blocks)
        self.query_builder = DCNInterFeatBuilder(args.nf)
        self.query_upsampler = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)

        self.dat_scale3 = CrossDeformableAttentionBlock(args.nf, args.nf)


    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        pass

    def forward(self, inp_dict):
        x0, x1, t, mean_ = self.normalize_w_rgb_mean(inp_dict['x0'], inp_dict['x1'], inp_dict['t'])
        feat0_1, feat0_2, feat0_3, feat0_4 = self.cnn_encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.cnn_encoder(x1)

        pred_feat_t_4, ft0_4, ft1_4 = self.query_builder(feat0_4, feat1_4)
        pred_feat_t_3 = self.query_upsampler(pred_feat_t_4)

        a = self.dat_scale3(pred_feat_t_3, feat0_3, ft0_4)






