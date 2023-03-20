import torch.nn as nn


class DeformAttn(nn.Module):
    def __init__(self, in_c, out_c, attn_window, groups, n_heads, mlp_ratio):
        super(DeformAttn, self).__init__()
        self.q_proj = nn.Linear(in_c, out_c, bias=True)
        self.k_proj = nn.Linear(in_c, out_c, bias=True)
        self.v_proj = nn.Linear(in_c, out_c, bias=True)

    def forward(self, x):
        pass