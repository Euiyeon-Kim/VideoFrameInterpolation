from modules.losses import *
from modules.warp import bwarp as warp


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :])
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :])
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1),
            convrelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1),
            convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1),
            convrelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1),
            convrelu(96, 96, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(192 + 1, 192),
            ResBlock(192, 32),
            nn.ConvTranspose2d(192, 76, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(220, 216),
            ResBlock(216, 32),
            nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(148, 144),
            ResBlock(144, 32),
            nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(100, 96),
            ResBlock(96, 32),
            nn.ConvTranspose2d(96, 8, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class IFRNet(nn.Module):
    def __init__(self, args):
        super(IFRNet, self).__init__()
        self.args = args

        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def forward(self, inp_dict):
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1, xt = x0 / 255., x1 / 255., xt / 255
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_

        # Generate multi-level features
        f0_1, f0_2, f0_3, f0_4 = self.encoder(x0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(x1)

        # Process decoder4 output
        out4 = self.decoder4(f0_4, f1_4, t)
        up_flow0_4, up_flow1_4 = out4[:, 0:2], out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        # Process decoder3 output
        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        # Process decoder2 output
        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        # Process decoder1 output
        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        # Final frame prediction
        x0_warp = warp(x0, up_flow0_1)
        x1_warp = warp(x1, up_flow1_1)
        imgt_merge = up_mask_1 * x0_warp + (1 - up_mask_1) * x1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        f01, f10 = inp_dict['f01'], inp_dict['f10']
        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)

        xt_ = xt - mean_
        ft_1, ft_2, ft_3, ft_4 = self.encoder(xt_)
        geo_loss = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))

        robust_weight0 = get_robust_weight(up_flow0_1, f01, beta=0.3)
        robust_weight1 = get_robust_weight(up_flow1_1, f10, beta=0.3)
        distill_loss = 0.01 * (self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - f01, weight=robust_weight0) + \
                               self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - f10, weight=robust_weight1) + \
                               self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - f01, weight=robust_weight0) + \
                               self.rb_loss(4.0 * resize(up_flow1_3, 4.0) - f10, weight=robust_weight1) + \
                               self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - f01, weight=robust_weight0) + \
                               self.rb_loss(8.0 * resize(up_flow1_4, 8.0) - f10, weight=robust_weight1))
        total_loss = l1_loss + census_loss + geo_loss + distill_loss

        return {
            'frame_preds': [imgt_pred],
            'xt_warp_x0': x0_warp + mean_,
            'xt_warp_x1': x1_warp + mean_,
            'x0_mask': up_mask_1,
            'f01': up_flow0_1,
            'f10': up_flow1_1,
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'flow_loss': distill_loss.item(),
            'geometry_loss': geo_loss.item(),
        }

