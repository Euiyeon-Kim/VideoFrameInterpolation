import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from models.IFRNet import convrelu
from models.IFRM2M import img_logger
from models.IFRM2M import Decoder32v1, Decoder1v1
from modules.warp import fwarp_mframes
from modules.positional_encoding import PositionEmbeddingSine
from modules.losses import Ternary, Charbonnier_Ada, Charbonnier_L1, get_robust_weight
from utils import normalize_imgnet, denormalize_imgnet_to01


class MultiScaleTridentConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            strides=1,
            paddings=0,
            dilations=1,
            dilation=1,
            groups=1,
            num_branch=1,
            test_branch_idx=-1,
            bias=False,
            norm=None,
            activation=None,
    ):
        super(MultiScaleTridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.dilation = dilation
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.strides = [_pair(stride) for stride in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        assert len({self.num_branch, len(self.paddings), len(self.strides)}) == 1

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, stride, padding, self.dilation, self.groups)
                for input, stride, padding in zip(inputs, self.strides, self.paddings)
            ]
        else:
            outputs = [
                F.conv2d(
                    inputs[0],
                    self.weight,
                    self.bias,
                    self.strides[self.test_branch_idx] if self.test_branch_idx == -1 else self.strides[-1],
                    self.paddings[self.test_branch_idx] if self.test_branch_idx == -1 else self.paddings[-1],
                    self.dilation,
                    self.groups,
                )
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 **kwargs,
                 ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride,
                                       norm_layer=norm_layer)  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim,
                                                      kernel_size=3,
                                                      strides=strides,
                                                      paddings=1,
                                                      num_branch=self.num_branch,
                                                      )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/8 or 1/4

        x = self.conv2(x)

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out = x

        return out


def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


def single_head_split_window_attention(q, k, v, num_splits=1, with_shift=False, h=None, w=None, attn_mask=None):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                          ) / scale_factor  # [B*K*K, H/K*W/K, H/K*W/K]

    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                       num_splits=num_splits, channel_last=True)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model=128, nhead=1, no_ffn=False, ffn_dim_expansion=4):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)      # [B, L, C]
        key = self.k_proj(key)          # [B, L, C]
        value = self.v_proj(value)      # [B, L, C]

        if attn_type == 'swin' and attn_num_splits > 1:  # self, cross-attn: both swin 2d
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask)
        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""
    def __init__(self, d_model=128, nhead=1, ffn_dim_expansion=4):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model, nhead=nhead, no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion)

        self.cross_attn_ffn = TransformerLayer(d_model=d_model, nhead=nhead,
                                               ffn_dim_expansion=ffn_dim_expansion)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_type=attn_type,
                                with_shift=with_shift,
                                attn_num_splits=attn_num_splits)

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                                     attn_type=attn_type,
                                     with_shift=with_shift,
                                     attn_num_splits=attn_num_splits)

        return source


def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature


def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w,
                                    shift_size_h, shift_size_w, device=torch.device('cuda')):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (slice(0, -window_size_h),
                slice(-window_size_h, -shift_size_h),
                slice(-shift_size_h, None))
    w_slices = (slice(0, -window_size_w),
                slice(-window_size_w, -shift_size_w),
                slice(-shift_size_w, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class FeatureTransformer(nn.Module):
    def __init__(self, num_layers=6, d_model=128, nhead=1, ffn_dim_expansion=4):
        super(FeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             ffn_dim_expansion=ffn_dim_expansion)
            for i in range(num_layers)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1, attn_type='swin', attn_num_splits=2, **kwargs):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        # 2d attention
        if 'swin' in attn_type and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]

        for i, layer in enumerate(self.layers):
            concat0 = layer(concat0, concat1,
                            height=h,
                            width=w,
                            attn_type=attn_type,
                            with_shift='swin' in attn_type and attn_num_splits > 1 and i % 2 == 1,
                            attn_num_splits=attn_num_splits,
                            shifted_window_attn_mask=shifted_window_attn_mask,
                            shifted_window_attn_mask_1d=None,
                            )

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return feature0, feature1


class SelfAttnPropagation(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, value):
        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        v = value.view(b, value.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, v)  # [B, H*W, 2]
        out = out.view(b, h, w, v.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def global_correlation_softmax(feature0, feature1):
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)                                 # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)                                                  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)     # [B, H, W, H, W]

    init_grid = coords_grid(b, h, w).to(correlation.device)                             # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)                                    # [B, H*  W, 2]

    correlation = correlation.view(b, h * w, h * w)                                     # [B, H*W, H*W]

    # For bi-directional flow
    correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)         # [2*B, H*W, H*W]
    init_grid = init_grid.repeat(2, 1, 1, 1)                                            # [2*B, 2, H, W]
    grid = grid.repeat(2, 1, 1)                                                         # [2*B, H*W, 2]
    b = b * 2

    prob = F.softmax(correlation, dim=-1)                                               # [B, H*W, H*W]
    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)      # [B, 2, H, W]
    flow = correspondence - init_grid

    f01, f10 = torch.chunk(flow, 2, dim=0)
    f01_prob, f10_prob = torch.chunk(prob, 2, dim=0)
    dual_prob = f01_prob * f10_prob.permute(0, 2, 1)

    return f01, f10, dual_prob


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 64, 3, 2, 1),
            convrelu(64, 64, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(64, 96, 3, 2, 1),
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


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor,
                         recompute_scale_factor=False, mode="bilinear", align_corners=True)


class GMM2Mv1(nn.Module):
    def __init__(self, args):
        super(GMM2Mv1, self).__init__()
        self.args = args
        self.n_branch = args.m2m_branch
        self.alpha = torch.nn.Parameter(10.0 * torch.ones(1, 1, 1, 1))

        self.nf = 128
        self.backbone = CNNEncoder(output_dim=self.nf, num_output_scales=1)
        self.transformer = FeatureTransformer(num_layers=6, d_model=self.nf, nhead=1, ffn_dim_expansion=4)

        self.context_encoder = Encoder()
        self.decoder3 = Decoder32v1(nc=128)
        self.decoder2 = Decoder32v1(nc=96)
        self.decoder1 = Decoder1v1(nc=64, n_branch=self.n_branch)

        self.tr_loss = Ternary(7)
        self.l1_loss = Charbonnier_L1()
        self.rb_loss = Charbonnier_Ada()

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        return img_logger(inp_dict, results_dict)

    def repeat_for_branch(self, tensor):
        _b, _c, _h, _w = tensor.shape
        return tensor.repeat(1, self.n_branch, 1, 1).view(_b, self.n_branch, _c, _h, _w).permute(1, 0, 2, 3, 4)

    def forward(self, inp_dict):
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']
        x0, x1 = normalize_imgnet(x0), normalize_imgnet(x1)
        t = t.unsqueeze(-1).unsqueeze(-1)
        nt = self.repeat_for_branch(t)

        # Extract feature from CNN
        org_feat0 = self.backbone(x0)
        org_feat1 = self.backbone(x1)
        b, _, fh, fw = org_feat0.shape

        # Attended feature
        feat0, feat1 = feature_add_position(org_feat0, org_feat1, 2, self.nf)
        feat0, feat1 = self.transformer(feat0, feat1, attn_num_splits=2)

        # 1 / 8 size optical flow estimation
        f01_4, f10_4, dual_prob = global_correlation_softmax(feat0, feat1)
        with torch.no_grad():
            x0_certainty = torch.max(dual_prob, dim=-1)[0].view(b, 1, fh, fw)
            x1_certainty = torch.max(dual_prob, dim=-2)[0].view(b, 1, fh, fw)
        z0_4 = (1. - x0_certainty) * self.alpha
        z1_4 = (1. - x1_certainty) * self.alpha

        feat0_1, feat0_2, feat0_3 = self.context_encoder(x0)
        feat1_1, feat1_2, feat1_3 = self.context_encoder(x1)

        f01_res_3, f10_res_3, z0_res_3, z1_res_3 = self.decoder3(feat0_3, feat1_3, f01_4, f10_4, z0_4, z1_4)
        f01_3 = 2.0 * resize(f01_4, scale_factor=2.0) + f01_res_3
        f10_3 = 2.0 * resize(f10_4, scale_factor=2.0) + f10_res_3
        z0_3, z1_3 = z0_res_3 + resize(z0_4, 2.0), z1_res_3 + resize(z1_4, 2.0)

        f01_res_2, f10_res_2, z0_res_2, z1_res_2 = self.decoder2(feat0_2, feat1_2, f01_3, f10_3, z0_3, z1_3)
        f01_2 = 2.0 * resize(f01_3, scale_factor=2.0) + f01_res_2
        f10_2 = 2.0 * resize(f10_3, scale_factor=2.0) + f10_res_2
        z0_2, z1_2 = z0_res_2 + resize(z0_3, 2.0), z1_res_2 + resize(z1_3, 2.0)

        f01_res_1, f10_res_1, z0_res_1, z1_res_1 = self.decoder1(feat0_1, feat1_1, f01_2, f10_2, z0_2, z1_2)
        f01_1 = (2.0 * resize(f01_2, scale_factor=2.0).unsqueeze(1) + f01_res_1).permute(1, 0, 2, 3, 4)
        f10_1 = (2.0 * resize(f10_2, scale_factor=2.0).unsqueeze(1) + f10_res_1).permute(1, 0, 2, 3, 4)
        z0_1, z1_1 = z0_res_1 + resize(z0_2, 2.0), z1_res_1 + resize(z1_2, 2.0)

        pred_xt, is_blank = fwarp_mframes(x0, f01_1 * nt, nt, x1, f10_1 * (1 - nt), (1 - nt),
                                          z0_1 * self.alpha, z1_1 * self.alpha)
        for_blank = (((1 - t) * x0 + t * x1) * is_blank * 1.0)
        imgt_pred = torch.clamp(denormalize_imgnet_to01(for_blank + pred_xt), 0, 1)

        if not self.training:
            return imgt_pred

        # Calculate loss
        f01, f10 = inp_dict['f01'], inp_dict['f10']
        xt = xt / 255.
        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)

        mu_f01_1, mu_f10_1 = torch.mean(f01_1, 0), torch.mean(f10_1, 0)
        robust_weight0 = get_robust_weight(mu_f01_1, f01, beta=0.3)
        robust_weight1 = get_robust_weight(mu_f01_1, f10, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(2.0 * resize(f01_2, 2.0) - f01, weight=robust_weight0) +
                               self.rb_loss(2.0 * resize(f10_2, 2.0) - f10, weight=robust_weight1) +
                               self.rb_loss(4.0 * resize(f01_3, 4.0) - f01, weight=robust_weight0) +
                               self.rb_loss(4.0 * resize(f10_3, 4.0) - f10, weight=robust_weight1) +
                               self.rb_loss(8.0 * resize(f01_4, 8.0) - f01, weight=robust_weight0) +
                               self.rb_loss(8.0 * resize(f10_4, 8.0) - f10, weight=robust_weight1))
        total_loss = l1_loss + census_loss + distill_loss

        return {
            'frame_preds': [imgt_pred],
            'x0_mask': z0_1,
            'x1_mask': z1_1,
            'f01': mu_f01_1,
            'f10': mu_f10_1,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'flow_loss': distill_loss.item(),
            'geometry_loss': 0.0,
            'alpha': self.alpha[0].item()
        }
