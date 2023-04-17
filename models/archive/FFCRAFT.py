import torch
import torch.nn as nn

import utils
from .BaseModel import GMBase
from modules.ffc import FFCTexture
from modules.refine import LAMAUpdateBlock


class FFCRAFT(GMBase):
    """
        Has convGRU for refinement
    """
    def __init__(self, args):
        super(FFCRAFT, self).__init__(args)
        self.refine_iters = args.refine_iters
        self.make_residual = args.make_residual
        self.refiner = LAMAUpdateBlock()
        self.cnet = FFCTexture(input_nc=3, n_downsampling=3, nfeats=(64, 128, 256, 256), n_blocks=4)

    def forward(self, x0, x1, xt, t):
        results_dict = {}
        final_preds = []
        t = t.unsqueeze(-1).unsqueeze(-1)

        x0_normed = utils.normalize_imgnet(x0)
        x1_normed = utils.normalize_imgnet(x1)

        feat0, feat1 = self.extract_cnn_feature(x0=x0_normed, x1=x1_normed)
        feat0, feat1 = self.get_cross_attended_feature(feat0, feat1)

        base_frame, log_base = self.generate_base_frame(feat0, feat1, x0, x1, t)
        results_dict.update(log_base)
        final_preds.append(base_frame)

        # Prepare context feature
        x0_01, x1_01 = x0 / 255., x1 / 255.
        cnet_inp = torch.concat((x0_01, x1_01), dim=1)
        cfeat = self.cnet(cnet_inp)

        net, inp = torch.split(cfeat, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        blended = base_frame.detach()
        for refine_idx in range(self.refine_iters):
            blended = blended.detach()
            cur_blended_feat = self.cnet(blended)
            net, residual = self.refiner(net, inp, cur_blended_feat)
            if self.make_residual:
                blended = torch.clamp(blended + torch.tanh(residual), 0, 1)
            else:
                blended = torch.sigmoid(residual)
            final_preds.append(blended)

        results_dict.update({
            'final_preds': final_preds,
            'alpha': self.alpha,
        })
        return results_dict

    def get_params_with_lr(self):
        return [
            {'params': self.backbone.parameters(), 'lr': self.args.small_lr},
            {'params': self.transformer.parameters(), 'lr': self.args.small_lr},
            {'params': self.upsampler.parameters(), 'lr': self.args.small_lr},
            {'params': self.feature_flow_attn.parameters(), 'lr': self.args.small_lr},
            {'params': self.alpha, 'lr': self.args.small_lr},
            {'params': self.refiner.parameters()},
            {'params': self.cnet.parameters()},
            {'params': self.inp_converter.parameters()},
            {'params': self.out_converter.parameters()},
        ]