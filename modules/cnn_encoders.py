import torch.nn as nn

from modules.basic_blocks import make_residual_layers


class SameChannelResEncoder(nn.Module):
    def __init__(self, nf, n_res_block):
        super(SameChannelResEncoder, self).__init__()
        layers = [
            nn.Conv2d(3, nf, 3, 1, 1, bias=True),
            nn.PReLU(nf),
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.PReLU(nf),
        ]
        if n_res_block > 0:
            layers.extend(make_residual_layers(nf=nf, n_layers=n_res_block))
        self.projection = nn.Sequential(*layers)
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
        feat1 = self.projection(x)
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        feat4 = self.fea_L4_conv(feat3)
        return feat1, feat2, feat3, feat4
