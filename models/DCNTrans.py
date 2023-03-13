import torch
import torch.nn as nn

from modules.losses import *
from models.GMTrans import Decoder2
from models.IFRNet import ResBlock, convrelu
from modules.residual_encoder import make_layer
from modules.dcnv2 import DeformableConv2d


class Decoder4v1(nn.Module):
    def __init__(self, nc):
        super(Decoder4v1, self).__init__()
        self.nc = nc
        self.convblock = nn.Sequential(
            convrelu(nc * 2, nc),
            ResBlock(nc, nc // 2),
        )
        self.dcn0t = DeformableConv2d(nc, nc)
        self.dcn1t = DeformableConv2d(nc, nc)
        self.blendblock = nn.Sequential(
            convrelu(nc * 2, nc),
            ResBlock(nc, nc // 2),
        )

    def forward(self, f0, f1):
        f01_offset_feat = self.convblock(torch.cat((f0, f1), 1))
        f10_offset_feat = self.convblock(torch.cat((f1, f0), 1))

        ft_from_f0, ft0_offset = self.dcn0t(f0, f01_offset_feat)
        ft_from_f1, ft1_offset = self.dcn1t(f1, f10_offset_feat)
        out = self.blendblock(torch.cat((ft_from_f0, ft_from_f1), 1))
        return out


class DCNTransv1(nn.Module):
    def __init__(self, args):
        super(DCNTransv1, self).__init__()
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
        self.decoder3 = Decoder4v1(nc=args.nf)

        self.query_builder2 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.decoder2 = Decoder2(inp_dim=args.nf, depth=6, num_heads=8, window_size=4)

        self.query_builder1 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.decoder1 = Decoder2(inp_dim=args.nf, depth=6, num_heads=4, window_size=4)

        self.prelu1 = nn.PReLU(args.nf)
        self.upconv1 = nn.Conv2d(args.nf, args.nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(args.nf, args.nf, 3, 1, 1)
        self.prelu2 = nn.PReLU(args.nf)
        self.conv_last = nn.Conv2d(args.nf, 3, 3, 1, 1)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        # self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0]/255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        return {
            'pred': pred_concat[0],
        }

    def extract_feature(self, x):
        feat1 = self.feature_extraction(self.conv_first(x))
        feat2 = self.fea_L2_conv(feat1)
        feat3 = self.fea_L3_conv(feat2)
        feat4 = self.fea_L4_conv(feat3)
        return feat2, feat3, feat4

    def generate_rgb_frame(self, feat, m):
        out = self.prelu1(self.pixel_shuffle(self.upconv1(feat)))
        out = self.prelu2(self.HRconv(out))
        out = self.conv_last(out)
        return torch.clamp(out + m, 0, 1)

    def forward(self, inp_dict):
        x0, x1, t = inp_dict['x0'], inp_dict['x1'], inp_dict['t']

        # Preprocess data
        t = t.unsqueeze(-1).unsqueeze(-1)
        x0, x1 = x0 / 255., x1 / 255.
        mean_ = (torch.mean(x0, dim=(2, 3), keepdim=True) + torch.mean(x1, dim=(2, 3), keepdim=True)) / 2
        x0_, x1_ = x0 - mean_, x1 - mean_

        feat0_1, feat0_2, feat0_3 = self.extract_feature(x0_)
        feat1_1, feat1_2, feat1_3 = self.extract_feature(x1_)

        # Pred with DCN
        pred_feat_t_3 = self.decoder3(feat0_3, feat1_3)

        # Attention
        feat_t_2_query = self.query_builder2(pred_feat_t_3)
        pred_feat_t_2 = self.decoder2(feat_t_2_query, feat0_2, feat1_2)

        feat_t_1_query = self.query_builder1(pred_feat_t_2)
        pred_feat_t_1 = self.decoder1(feat_t_1_query, feat0_1, feat1_1)

        imgt_pred = self.generate_rgb_frame(pred_feat_t_1, mean_)

        if not self.training:
            return imgt_pred

        # Calculate loss on training phase
        # f01, f10 = inp_dict['f01'], inp_dict['f10']
        xt = inp_dict['xt'] / 255
        xt_ = xt - mean_
        _, _, ft_3 = self.extract_feature(xt_)

        l1_loss = self.l1_loss(imgt_pred - xt)
        census_loss = self.tr_loss(imgt_pred, xt)
        geo_loss = 0.01 * (self.gc_loss(pred_feat_t_3, ft_3))
        total_loss = l1_loss + census_loss + geo_loss

        return {
            'frame_preds': [imgt_pred],
        }, total_loss, {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
            'geometry_loss': geo_loss.item(),
        }
