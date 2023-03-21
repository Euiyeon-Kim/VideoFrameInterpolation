import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformAttn(nn.Module):
    def __init__(self, in_c, out_c, n_samples, n_groups, n_heads):
        super(DeformAttn, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.q_proj = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.k_proj = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.v_proj = nn.Conv2d(in_c, out_c, 1, 1, 0)

        self.n_groups = n_groups
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


