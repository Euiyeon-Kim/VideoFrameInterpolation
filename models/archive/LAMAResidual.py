import torch
import torch.nn as nn

import utils
from .BaseModel import GMBase
from modules.ffc import FFCResNetGenerator
from modules.matching import global_correlation_softmax, align_features_for_t
from modules.geometry import forward_backward_consistency_check, compute_out_of_boundary_mask


class LAMAResidual(GMBase):
    """
        Has convGRU for refinement
    """
    def __init__(self, args):
        super(LAMAResidual, self).__init__(args)
        self.generator = FFCResNetGenerator(input_nc=3*3, output_nc=3, add_out_act=True)

    def forward(self, x0, x1, xt, t):
        results_dict = {}
        final_preds = []

        t = t.unsqueeze(-1).unsqueeze(-1)
        _, _, h, w = x0.shape
        feat0_origin, feat1_origin = self.extract_cnn_feature(x0, x1)
        feat0, feat1 = self.get_cross_attended_feature(feat0_origin, feat1_origin)
        b, _, fh, fw = feat0.shape

        corr_flow_preds, dual_probs = global_correlation_softmax(feat0, feat1, pred_bidir_flow=True)

        # Self attention for flow refine
        feats = torch.cat((feat0, feat1), dim=0)  # [2*B, C, H, W] for propagation
        flow_preds = self.feature_flow_attn(feats, corr_flow_preds.detach())
        f01, f10 = torch.chunk(flow_preds, 2, dim=0)

        # Make upsample mask
        _for_mask = torch.cat((torch.cat((f01, feat0), dim=1), torch.cat((f10, feat1), dim=1)), dim=0)
        mask = self.upsampler(_for_mask)

        # Get origin size flow
        flow_up = self.upsample_with_mask(flow_preds, mask, 8) * self.upsample_factor
        f01_up, f10_up = torch.chunk(flow_up, 2, dim=0)

        with torch.no_grad():
            # Get dual softmax certainty
            # Higher -> Has match
            matching_certainty = torch.max(dual_probs.detach(), dim=-1)[0].view(2 * b, 1, fh, fw)
            # x0_certainty_small, x1_certainty_small = torch.chunk(matching_certainty, 2)
            certainty_up = self.upsample_with_mask(matching_certainty, mask, 8)
            x0_certainty, x1_certainty = torch.chunk(certainty_up, 2)

        # How much to reflect certainty
        z0 = x0_certainty * self.alpha
        z1 = x1_certainty * self.alpha

        xt_fwarp_x0 = self.fwarper(x0, f01_up * t, z0)
        xt_fwarp_x1 = self.fwarper(x1, f10_up * (1 - t), z1)

        zt_fwarp_z0 = self.fwarper(x0_certainty, f01_up * t, z0)
        zt_fwarp_z1 = self.fwarper(x1_certainty, f10_up * (1 - t), z1)

        fwd_occ, bwd_occ = forward_backward_consistency_check(f01, f10, alpha=0.01, beta=0.5)
        occ_up = self.upsample_with_mask(torch.concat((fwd_occ, bwd_occ), dim=0), mask, 8)
        fwd_occ_up, bwd_occ_up = torch.chunk(occ_up, 2, dim=0)

        occ_fwarp_fwd_occ = self.fwarper(fwd_occ_up, f01_up * t, z0)
        occ_fwarp_bwd_occ = self.fwarper(bwd_occ_up, f10_up * (1 - t), z1)

        x0_mask = zt_fwarp_z0 * (1 - zt_fwarp_z1) * (1 - occ_fwarp_bwd_occ) + 1e-6
        x1_mask = zt_fwarp_z1 * (1 - zt_fwarp_z0) * (1 - occ_fwarp_fwd_occ) + 1e-6
        blended = (xt_fwarp_x0 * x0_mask + xt_fwarp_x1 * x1_mask) / (x0_mask + x1_mask)
        blended_01 = utils.denorm_to_01(blended)
        final_preds.append(blended_01)

        generator_inp = torch.concat((x0, blended.detach(), x1), dim=1)
        residual = self.generator(generator_inp)
        final = torch.clamp(utils.denorm_to_01(blended + residual), 0, 1)
        final_preds.append(final)

        results_dict.update({
            'final_preds': final_preds,
            'xt_warp_x0': xt_fwarp_x0,
            'xt_warp_x1': xt_fwarp_x1,
            'x0_mask': x0_mask,
            'x1_mask': x1_mask,
            'alpha': self.alpha,
            'f01': f01_up,
            'f10': f10_up,
        })
        return results_dict

    def get_params_with_lr(self):
        return [
            {'params': self.backbone.parameters(), 'lr': self.args.small_lr},
            {'params': self.transformer.parameters(), 'lr': self.args.small_lr},
            {'params': self.upsampler.parameters(), 'lr': self.args.small_lr},
            {'params': self.feature_flow_attn.parameters(), 'lr': self.args.small_lr},
            {'params': self.alpha, 'lr': self.args.small_lr},
            {'params': self.generator.parameters()}
        ]