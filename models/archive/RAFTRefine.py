import torch
import torch.nn as nn

import utils
import modules.losses as losses
from modules.bwarp import warp
from modules.fwarp import SoftSplat
from modules.attention import SelfAttnPropagation
from modules.refine import GRUforFeat, UpConvHead
from modules.transformer import FeatureTransformer
from modules.swin_utils import feature_add_position
from modules.matching import global_correlation_softmax
from modules.geometry import forward_backward_consistency_check, compute_out_of_boundary_mask


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.PReLU(planes)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d, num_output_scales=1, **kwargs):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # # highest resolution 1/4 or 1/8
        # stride = 2 if num_output_scales == 1 else 1
        # self.layer3 = self._make_layer(feature_dims[2], stride=stride,
        #                                norm_layer=norm_layer)  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[1], output_dim, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        # x = self.layer3(x)  # 1/8 or 1/4

        x = self.conv2(x)
        return x


class RAFTRefine(nn.Module):
    """
        Has convGRU for refinement
    """
    def __init__(self, args):
        super(RAFTRefine, self).__init__()
        self.args = args
        self.gamma = args.gamma
        self.w_mode = args.warp_mode
        self.train_alpha = args.train_alpha
        self.refine_iters = args.refine_iters
        self.upsample_factor = args.upsample_factor

        self.fwarper = SoftSplat()
        self.alpha = nn.Parameter(torch.tensor([args.init_alpha]), requires_grad=args.train_alpha)

        # GMFlow
        self.encoder = CNNEncoder()
        self.transformer = FeatureTransformer(num_layers=args.nlayer, d_model=args.nf,
                                              nhead=args.nhead, ffn_dim_expansion=args.ffc_expansion)
        self.feature_flow_attn = SelfAttnPropagation(in_channels=args.nf)

        self.tr_loss = losses.Ternary(7)
        self.gc_loss = losses.Geometry(3)
        self.l1_loss = losses.Charbonnier_L1()
        self.rb_loss = losses.Charbonnier_Ada()

    def get_params_with_lr(self):
        params = [
            {'params': self.encoder.parameters(), 'lr': self.args.small_lr},
            # {'params': self.transformer.parameters(), 'lr': self.args.small_lr},
            # {'params': self.upsampler.parameters(), 'lr': self.args.small_lr},
            # {'params': self.feature_flow_attn.parameters(), 'lr': self.args.small_lr},
        ]
        if self.train_alpha:
            params = params + [{'params': self.alpha, 'lr': self.args.small_lr}]
        return params

    def calcul_losses(self, results, xt):
        final_preds = results['final_preds']
        n_predictions = len(final_preds)
        gt_01 = xt / 255.

        l1_loss, census_loss, feat_geo_loss = 0.0, 0.0, 0.0
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_l1_loss = self.charbonnier(final_preds[i] - gt_01)
            l1_loss += i_weight * i_l1_loss
            i_census_loss = self.census(final_preds[i], gt_01).mean()
            census_loss += i_weight * i_census_loss

        total = l1_loss + census_loss
        metric = {}
        metric.update({
            'total_loss': total.item(),
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item() if self.use_census else 0.0,
            'alpha': self.alpha[0].item(),
        })

        return total, metric

    def get_bidirectional_flow(self, feat0, feat1):
        b, _, fh, fw = feat0.shape
        corr_flow_preds, dual_probs = global_correlation_softmax(feat0, feat1, pred_bidir_flow=True)

        # Self attention for flow refine
        feats = torch.cat((feat0, feat1), dim=0)  # [2*B, C, H, W] for propagation
        flow_preds = self.feature_flow_attn(feats, corr_flow_preds.detach())
        f01, f10 = torch.chunk(flow_preds, 2, dim=0)

        # Calculate importance metric
        with torch.no_grad():
            matching_certainty = torch.max(dual_probs.detach(), dim=-1)[0].view(2 * b, 1, fh, fw)
            small_z0, small_z1 = torch.chunk(matching_certainty, 2)
            certainty_up = self.upsample_with_mask(matching_certainty, mask, 8)
            x0_certainty, x1_certainty = torch.chunk(certainty_up, 2)

        small_z0, small_z1 = small_z0 * self.alpha, small_z1 * self.alpha
        z0, z1 = x0_certainty * self.alpha, x1_certainty * self.alpha

        small_ft0 = -self.fwarper(f01, f01 * t, small_z0) * t
        small_ft0_mask = compute_out_of_boundary_mask(small_ft0).unsqueeze(1)
        small_ft0 = small_ft0 * small_ft0_mask

        small_ft1 = -self.fwarper(f10, f10 * (1 - t), small_z1) * (1 - t)
        small_ft1_mask = compute_out_of_boundary_mask(small_ft1).unsqueeze(1)
        small_ft1 = small_ft1 * small_ft1_mask

        # Warp frame
        if self.w_mode == 'f':
            xt_warp_x0, xt_warp_x1 = self.fwarp_frame(x0, x1, f01_up, f10_up, t, z0, z1)
        else:
            xt_warp_x0, xt_warp_x1, ft0_out_mask, ft1_out_mask = self.bwarp_frame(x0, x1, f01_up, f10_up, t, z0, z1)

        # Calculate blending mask
        zt_fwarp_z0 = self.fwarper(x0_certainty, f01_up * t, z0)
        zt_fwarp_z1 = self.fwarper(x1_certainty, f10_up * (1 - t), z1)

        fwd_occ, bwd_occ = forward_backward_consistency_check(f01, f10, alpha=0.01, beta=0.5)
        occ_up = self.upsample_with_mask(torch.concat((fwd_occ, bwd_occ), dim=0), mask, 8)
        fwd_occ_up, bwd_occ_up = torch.chunk(occ_up, 2, dim=0)

        occ_fwarp_fwd_occ = self.fwarper(fwd_occ_up, f01_up * t, z0)
        occ_fwarp_bwd_occ = self.fwarper(bwd_occ_up, f10_up * (1 - t), z1)

        x0_mask = zt_fwarp_z0 * (1 - zt_fwarp_z1) * (1 - occ_fwarp_bwd_occ)
        x1_mask = zt_fwarp_z1 * (1 - zt_fwarp_z0) * (1 - occ_fwarp_fwd_occ)
        if self.w_mode == 'b':
            x0_mask = x0_mask * ft0_out_mask
            x1_mask = x1_mask * ft1_out_mask

        x0_mask, x1_mask = x0_mask + 1e-6, x1_mask + 1e-6
        blended = (xt_warp_x0 * x0_mask + xt_warp_x1 * x1_mask) / (x0_mask + x1_mask)
        return blended, small_ft0, small_ft1, {
            'xt_warp_x0': xt_warp_x0,
            'xt_warp_x1': xt_warp_x1,
            'z0': z0,
            'z1': z1,
            'x0_mask': x0_mask,
            'x1_mask': x1_mask,
            'f01': f01_up,
            'f10': f10_up,
        }

    def get_cross_attended_feature(self, feat0, feat1):
        feat0, feat1 = feature_add_position(feat0, feat1, 2, self.args.nf)
        feat0, feat1 = self.transformer(feat0, feat1, attn_num_splits=2)
        return feat0, feat1

    def forward(self, inp_dict):
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']
        x0, x1, xt = x0 / 255., x1 / 255., xt / 255
        t = t.unsqueeze(-1).unsqueeze(-1)

        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1, xt = x0 - mean_, x1 - mean_, xt - mean_

        org_feat0, org_feat1 = torch.chunk(self.encoder(torch.cat((x0, x1), dim=0)), 2, dim=0)
        feat0, feat1 = self.get_cross_attended_feature(org_feat0, org_feat1)

        base_frame, small_ft0, small_ft1, log_base = self.get_bidirectional_flow(feat0, feat1)
        results_dict.update(log_base)
        final_preds.append(base_frame)

        feat_t_from_0 = self.bwarper(org_feat0, small_ft0)
        feat_t_from_1 = self.bwarper(org_feat1, small_ft1)

        net_inp = self.concat_to_inp_proj(torch.concat((feat_t_from_0, feat_t_from_1), dim=1))
        net, inp = torch.chunk(net_inp, chunks=2, dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        feat_t = self.concat_to_feat_proj(torch.concat((feat_t_from_0, feat_t_from_1), dim=1))
        feat_preds.append(feat_t)
        for refine_idx in range(self.refine_iters):
            feat_t = feat_t.detach()
            net, res_feat_t = self.refiner(net, inp, feat_t)
            feat_t = feat_t + res_feat_t
            feat_preds.append(feat_t)
            pred = torch.sigmoid(self.upconv(feat_t))
            final_preds.append(pred)

        results_dict.update({
            'feat_preds': feat_preds,
            'final_preds': final_preds,
            'alpha': self.alpha,
        })
        return results_dict
