import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import modules.losses as losses
from models.IFRNet import convrelu
from utils.flow_viz import flow_tensor_to_np
from modules.warp import fwarp_using_two_frames
from models.GMM2M import FeatureTransformer, feature_add_position, global_correlation_softmax, resize, CNNEncoder
from utils import normalize_imgnet, denormalize_imgnet_to01


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 48, 3, 2, 1),
            convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(48, 96, 3, 2, 1),
            convrelu(96, 96, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(96, 128, 3, 2, 1),
            convrelu(128, 128, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        return f1, f2, f3


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv, mask=None):
        B_, N, C = q.shape
        q = self.q_proj(q).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv_proj(kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows     # nW*B, W, W, C


def window_reverse(windows, window_size, B, H, W):
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x            # B, H, W, C


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class SwinIRBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.merge = nn.Linear(dim, dim, bias=False)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, feat, attn_mask):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)               # b, h, w, c
        feat = feat.permute(0, 2, 3, 1)

        window_size, shift_size = get_window_size((h, w), (self.window_size, self.window_size),
                                                  (self.shift_size, self.shift_size))

        shortcut = x
        pad_h = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_w = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        feat = F.pad(feat, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            shifted_feat = torch.roll(feat, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask_x = attn_mask
        else:
            shifted_x = x
            shifted_feat = feat
            attn_mask_x = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)                        # nW*B, window_size, window_size, C
        feat_windows = window_partition(shifted_feat, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], c)          # nW*B, window_size*window_size, C
        feat_windows = feat_windows.view(-1, window_size[0] * window_size[1], c)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, feat_windows, mask=attn_mask_x)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], c)
        shifted_x = window_reverse(attn_windows, window_size, b, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_w > 0 or pad_h > 0:
            x = x[:, :, :h, :w, :].contiguous()

        x = self.merge(x)
        x = self.norm1(x)

        # FFN
        x = shortcut + x
        x = x + self.norm2(self.mlp(x))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinIRBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, norm_layer=norm_layer
            )
            for i in range(depth)]
        )
        self.mixer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim*2, dim, 3, 1, 1, 1, 1, bias=True),
                nn.PReLU(dim)
            )
            for i in range(depth)]
        )

    def calculate_mask(self, x):
        b, c, h, w = x.shape
        window_size, shift_size = get_window_size((h, w), (self.window_size, self.window_size),
                                                  (self.shift_size, self.shift_size))

        Hp = int(np.ceil(h / window_size[0])) * window_size[0]
        Wp = int(np.ceil(w / window_size[1])) * window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 H W 1

        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, (self.window_size, self.window_size))
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, source, target):
        # x: b, c, h, w
        attn_mask = self.calculate_mask(x)
        for idx, blk in enumerate(self.blocks):
            source_attended_x = blk(x, source, attn_mask)       # b, h, w, c
            target_attended_x = blk(x, target, attn_mask)
            mix_layer = self.mixer[idx]
            x = mix_layer(torch.cat((source_attended_x, target_attended_x), dim=-1).permute(0, 3, 1, 2))
        return x


class Decoder3(nn.Module):
    def __init__(self, inp_dim, out_dim, depth, num_heads, window_size, mlp_ratio=2.):
        super(Decoder3, self).__init__()
        self.transformer = BasicLayer(dim=inp_dim, depth=depth, num_heads=num_heads,
                                      window_size=window_size, mlp_ratio=mlp_ratio)
        self.upconv = nn.ConvTranspose2d(inp_dim, out_dim, 4, 2, 1, bias=True)

    def forward(self, x, source, target):
        x = self.transformer(x, source, target)
        return self.upconv(x)


class Decoder2(nn.Module):
    def __init__(self, inp_dim, depth, num_heads, window_size, mlp_ratio=4.):
        super(Decoder2, self).__init__()
        self.transformer = BasicLayer(dim=inp_dim, depth=depth, num_heads=num_heads,
                                      window_size=window_size, mlp_ratio=mlp_ratio)
        self.transformer.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, source, target):
        x = self.transformer(x, source, target)
        return x


class GMTransv1(nn.Module):
    def __init__(self, args):
        super(GMTransv1, self).__init__()
        self.args = args
        self.alpha = torch.nn.Parameter(10.0 * torch.ones(1, 1, 1, 1))

        self.nf = 128
        # self.backbone = CNNEncoder(output_dim=self.nf, num_output_scales=1)
        self.transformer = FeatureTransformer(num_layers=6, d_model=self.nf, nhead=1, ffn_dim_expansion=4)

        self.encoder = Encoder()
        self.decoder4 = nn.ConvTranspose2d(self.nf, 96, 4, 2, 1, bias=True)
        self.decoder3 = Decoder3(inp_dim=96, out_dim=48, depth=6, num_heads=6, window_size=4)
        self.decoder2 = Decoder2(inp_dim=48, depth=6, num_heads=6, window_size=4)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(48, 3 * 4, 3, 1, 1, 1, 1, bias=True),
            nn.PixelShuffle(2)
        )

        self.l1_loss = losses.Charbonnier_L1()
        self.tr_loss = losses.Ternary(7)
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0] / 255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]
        residual = torch.clamp(results_dict['frame_preds'][1][0][None], 0, 1)
        base = torch.clamp(results_dict['frame_preds'][0][0][None], 0, 1)

        fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
        viz_flow = torch.cat((x0_01, torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), x1_01), dim=-1)

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        process_concat = torch.cat((base, residual, pred_last, xt_01[None]), dim=-1)
        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
            'process': process_concat[0],
        }

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        # x0, x1 = normalize_imgnet(x0), normalize_imgnet(x1)
        x0, x1 = x0 / 255., x1 / 255.
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_

        # Encode CNN feature
        feat0_1, feat0_2, feat0_3 = self.encoder(x0)
        feat1_1, feat1_2, feat1_3 = self.encoder(x1)
        # feat0_3 = self.backbone(x0)
        # feat1_3 = self.backbone(x1)
        b, _, fh, fw = feat0_3.shape

        # Attended feature
        attended_feat0_3, attended_feat1_3 = feature_add_position(feat0_3, feat1_3, 2, self.nf)
        attended_feat0_3, attended_feat1_3 = self.transformer(attended_feat0_3, attended_feat1_3, attn_num_splits=2)

        f01_4, f10_4, dual_prob = global_correlation_softmax(attended_feat0_3, attended_feat1_3)
        with torch.no_grad():
            x0_certainty = torch.max(dual_prob, dim=-1)[0].view(b, 1, fh, fw)
            x1_certainty = torch.max(dual_prob, dim=-2)[0].view(b, 1, fh, fw)
        z0_4 = (1. - x0_certainty) * self.alpha
        z1_4 = (1. - x1_certainty) * self.alpha

        # Make feat_t_4
        pred_feat_t_4, pred_feat_t_4_is_blank = fwarp_using_two_frames(feat0_3, f01_4 * t, t,
                                                                       feat1_3, f10_4 * (1 - t), (1 - t),
                                                                       z0_4, z1_4)

        # Make feat_t_3
        pred_feat_t_3 = self.decoder4(pred_feat_t_4)
        pred_feat_t_2 = self.decoder3(pred_feat_t_3, feat0_2, feat1_2)
        pred_feat_t_1 = self.decoder2(pred_feat_t_2, feat0_1, feat1_1)
        residual = self.decoder1(pred_feat_t_1)

        pred_f01 = resize(f01_4, 8.) * 8.
        pred_f10 = resize(f10_4, 8.) * 8.
        z0 = resize(z0_4, 8.)
        z1 = resize(z1_4, 8.)
        img_t_base, _ = fwarp_using_two_frames(x0, pred_f01 * t, t,
                                               x1, pred_f10 * (1 - t), (1 - t), z0, z1)
        imgt_pred = torch.clamp(denormalize_imgnet_to01(img_t_base + residual + mean_), 0, 1)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        f01, f10 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        img_base = torch.clamp(img_t_base + mean_, 0, 1)
        l1_loss = self.l1_loss(imgt_pred - xt) + self.l1_loss(img_base - xt)
        census_loss = self.tr_loss(imgt_pred, xt) + self.tr_loss(img_base, xt)

        distill_loss = 0.01 * (self.mse_loss(pred_f01, f01) + self.mse_loss(pred_f10, f10))
        total_loss = l1_loss + census_loss + distill_loss

        return {
            'frame_preds': [img_t_base + mean_, residual + mean_, imgt_pred],
            'f01': pred_f01,
            'f10': pred_f10,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'flow_loss': distill_loss.item(),
        }
