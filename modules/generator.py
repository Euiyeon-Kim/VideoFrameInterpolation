import torch
import torch.nn as nn

from .basic_blocks import make_residual_layers


class BasicResPixelShuffleGenerator(nn.Module):
    def __init__(self, nf, num_res_blocks):
        super(BasicResPixelShuffleGenerator, self).__init__()
        self.nf = nf
        self.reconstruction = nn.Sequential(*make_residual_layers(nf=nf, n_layers=num_res_blocks))
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU(nf)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.prelu2 = nn.PReLU(nf)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, feat, m):
        out = self.reconstruction(feat)
        out = self.prelu1(self.pixel_shuffle(self.upconv1(out)))
        out = self.prelu2(self.HRconv(out))
        out = self.conv_last(out)
        return torch.clamp(out + m, 0, 1)