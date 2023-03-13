import torch
import torch.nn as nn

from models.IFRNet import ResBlock, convrelu
from modules.residual_encoder import make_layer


class Decoder4v1(nn.Module):
    def __init__(self, nc):
        super(Decoder4v1, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            convrelu(nc * 2, nc * 2),
            ResBlock(nc * 2, nc),
        )

    def forward(self, f0, f1):
        f01_offset = self.convblock(torch.cat([f0, f1], 1))
        f10_offset = self.convblock(torch.cat([f1, f0], 1))

        return f01_offset


class DCNIFRv1(nn.Module):
    def __init__(self, args):
        super(DCNIFRv1, self).__init__()
        self.args = args

        # Feature extraction
        layers = []
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.feature_extraction = make_layer(nf=args.nf, n_layers=args.enc_res_blocks)
        self.fea_L2_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.fea_L3_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.fea_L4_conv = nn.Sequential(
            nn.Conv2d(args.nf, args.nf, 3, 2, 1, bias=True),
            nn.PReLU(args.nf),
            nn.Conv2d(args.nf, args.nf, 3, 1, 1, bias=True),
            nn.PReLU(args.nf),
        )
        self.decoder4 = Decoder4v1(nc=args.nf)

    def extract_feature(self, x):
        feat1 = self.feature_extraction(self.conv_first(x))
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        feat4 = self.fea_L4_conv(feat3)
        return feat1, feat2, feat3, feat4

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1 = x0 / 255., x1 / 255.
        mean_ = (torch.mean(x0, dim=(2, 3), keepdim=True) + torch.mean(x1, dim=(2, 3), keepdim=True)) / 2
        x0_, x1_ = x0 - mean_, x1 - mean_

        feat0_1, feat0_2, feat0_3, feat0_4 = self.extract_feature(x0_)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.extract_feature(x1_)

        self.decoder4(feat0_4, feat1_4)

        print(feat0_3.shape)
        exit()
