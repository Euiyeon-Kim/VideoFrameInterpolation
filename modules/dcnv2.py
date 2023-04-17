import torch
import torch.nn as nn
import torchvision.ops

from modules.warp import bwarp


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.group = groups

        # Hopefully close to Optical Flow
        # self.offset_flow_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     nn.PReLU(in_channels),
        #     nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     nn.PReLU(in_channels),
        #     nn.Conv2d(in_channels, 2, 3, 1, 1),
        # )
        self.offset_flow_conv = nn.Conv2d(in_channels, 2, 3, 1, 1)

        # Generate residual flow and modulator
        self.conv_offset_mask = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 2, in_channels, 3, 1, 1),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, groups * 3 * kernel_size * kernel_size, 3, 1, 1),
        )
        # self.conv_offset_mask = nn.Conv2d(in_channels * 2 + 2, groups * 3 * kernel_size * kernel_size, 3, 1, 1)
        self.conv_offset_mask.apply(self.init_zero)

        self.regular_conv = nn.Conv2d(in_channels=in_channels // groups,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=padding)

    @staticmethod
    def init_zero(m):
        if type(m) == nn.Conv2d:
            m.weight.data.zero_()
            m.bias.data.zero_()

    def forward(self, x, movement_feat):
        offset_flow_tx = self.offset_flow_conv(movement_feat)   # B, 2, fH, fW
        feat_t_from_x = bwarp(x, offset_flow_tx)

        out = self.conv_offset_mask(torch.cat((feat_t_from_x, movement_feat, offset_flow_tx), dim=1))
        res_o1, res_o2, mask = torch.chunk(out, 3, dim=1)
        res_offset = 2. * torch.tanh(torch.cat((res_o1, res_o2), dim=1))

        offset = res_offset + offset_flow_tx.flip(1).repeat(1, res_offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x, offset_flow_tx
