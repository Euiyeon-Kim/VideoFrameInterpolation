import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def coords_grid(b, h, w, device=None):
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, -1, w, -1)
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, -1, -1, h)
    grid = torch.cat([xx, yy], 1).to(device).unsqueeze(1)
    return grid


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.apply(init_weights)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class InterFrameAttention(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=True, qk_scale=None, mlp_ratio=4.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(2, motion_dim, bias=qkv_bias)

        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

        self.apply(init_weights)

    def forward(self, feat0, feat1):
        _, _, H, W = feat0.shape
        feat0 = rearrange(feat0, 'b c h w -> b (h w) c')
        feat0 = self.norm1(feat0)
        feat1 = rearrange(feat1, 'b c h w -> b (h w) c')
        feat1 = self.norm1(feat1)
        feat_s = torch.cat((feat0, feat1), dim=0)
        feat_t = torch.cat((feat1, feat0), dim=0)

        B, N, C = feat_s.shape
        coords = coords_grid(B, H, W, feat_s.device)
        cor = rearrange(coords, 'b c h w -> b (h w) c')

        q = self.q_proj(feat_s).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(feat_t).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cor_embed_ = self.cor_embed(cor)
        cor_embed = cor_embed_.reshape(B, N, self.num_heads, self.motion_dim // self.num_heads).permute(0, 2, 1, 3)

        # Calculate score
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        feat_s = x + feat_s

        out = feat_s + self.mlp(self.norm2(feat_s), H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

        # Attention score to motion
        c_reverse = (attn @ cor_embed).transpose(1, 2).reshape(B, N, -1)
        motion = self.motion_proj(c_reverse - cor_embed_)
        motion = rearrange(motion, 'b (h w) c -> b c h w', h=H, w=W)

        return out, motion
