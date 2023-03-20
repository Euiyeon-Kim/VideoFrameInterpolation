import os
import math

import torch
from torch import nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from einops.layers.torch import Rearrange
from distutils.version import LooseVersion
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
deform_attn_ext = load(
    'deform_attn',
    sources=[
        os.path.join(module_path, 'ops', 'deform_attn_ext.cpp'),
        os.path.join(module_path, 'ops', 'deform_attn_cuda_pt110.cpp' if LooseVersion(torch.__version__) >= LooseVersion(
            '1.10.0') else 'deform_attn_cuda_pt109.cpp'),
        os.path.join(module_path, 'ops', 'deform_attn_cuda_kernel.cu')]
)


class DeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, q, kv, offset, kernel_h, kernel_w, stride=1, padding=0, dilation=1,
                attention_heads=1, deformable_groups=1, clip_size=1):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.attention_heads = attention_heads
        ctx.deformable_groups = deformable_groups
        ctx.clip_size = clip_size
        if q.requires_grad or kv.requires_grad or offset.requires_grad:
            ctx.save_for_backward(q, kv, offset)
        output = q.new_empty(q.shape)
        ctx._bufs = [q.new_empty(0), q.new_empty(0), q.new_empty(0), q.new_empty(0), q.new_empty(0)]
        deform_attn_ext.deform_attn_forward(q, kv, offset, output,
                                            ctx._bufs[0], ctx._bufs[1], ctx._bufs[2], ctx.kernel_h, ctx.kernel_w,
                                            ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
                                            ctx.attention_heads, ctx.deformable_groups, ctx.clip_size)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        q, kv, offset = ctx.saved_tensors
        grad_q = torch.zeros_like(q)
        grad_kv = torch.zeros_like(kv)
        grad_offset = torch.zeros_like(offset)
        deform_attn_ext.deform_attn_backward(q, kv, offset, ctx._bufs[0], ctx._bufs[1], ctx._bufs[2], ctx._bufs[3], ctx._bufs[4],
                                                       grad_q, grad_kv, grad_offset,
                                                       grad_output, ctx.kernel_h, ctx.kernel_w, ctx.stride,
                                                       ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
                                                       ctx.attention_heads, ctx.deformable_groups, ctx.clip_size)
        return (grad_q, grad_kv, grad_offset, None, None, None, None, None, None, None, None)


deform_attn = DeformAttnFunction.apply


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DeformAttn(nn.Module):
    def __init__(self, in_c, out_c, attention_window=(3, 3), deformable_groups=12, attention_heads=12, mlp_ratio=2):
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_h = attention_window[0]
        self.kernel_w = attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.deformable_groups = deformable_groups
        self.attention_heads = attention_heads
        self.clip_size = 1
        self.stride = 1
        self.padding = self.kernel_h//2
        self.dilation = 1

        self.proj_q = nn.Sequential(Rearrange('n c h w -> n h w c'),
                                    nn.Linear(self.in_channels, self.in_channels),
                                    Rearrange('n h w c -> n c h w'))
        self.proj_k = nn.Sequential(Rearrange('n c h w -> n h w c'),
                                    nn.Linear(self.in_channels, self.in_channels),
                                    Rearrange('n h w c -> n c h w'))
        self.proj_v = nn.Sequential(Rearrange('n c h w -> n h w c'),
                                    nn.Linear(self.in_channels, self.in_channels),
                                    Rearrange('n h w c -> n c h w'))
        self.mlp = nn.Sequential(Rearrange('n c h w -> n h w c'),
                                 Mlp(self.in_channels, self.in_channels * mlp_ratio),
                                 Rearrange('n h w c -> n c h w'))

    def forward(self, q, k, v, offset):
        q = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups, self.clip_size)
        v = v + self.mlp(v)
        return v
