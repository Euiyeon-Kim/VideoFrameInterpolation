import torch
import torch.nn as nn

from utils.flow_viz import flow_tensor_to_np
from modules.residual_encoder import make_layer


class Basemodel(nn.Module):
    def __init__(self, args):
        super(Basemodel, self).__init__()
        self.args = args
        self.reconstruction = nn.Sequential(*make_layer(nf=args.nf, n_layers=args.dec_res_blocks))
        self.upconv1 = nn.Conv2d(args.nf, args.nf * 4, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU(args.nf)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(args.nf, args.nf, 3, 1, 1)
        self.prelu2 = nn.PReLU(args.nf)
        self.conv_last = nn.Conv2d(args.nf, 3, 3, 1, 1)

    def generate_rgb_frame(self, feat, m):
        out = self.reconstruction(feat)
        out = self.prelu1(self.pixel_shuffle(self.upconv1(out)))
        out = self.prelu2(self.HRconv(out))
        out = self.conv_last(out)
        return torch.clamp(out + m, 0, 1)

    @staticmethod
    def get_log_dict(inp_dict, results_dict):
        x0, x1, xt = inp_dict['x0'], inp_dict['x1'], inp_dict['xt']
        x0_01, x1_01, xt_01 = x0[0] / 255., x1[0] / 255., xt[0] / 255.
        pred_last = results_dict['frame_preds'][-1][0][None]

        half = (x0_01 + x1_01) / 2
        err_map = (xt_01 - pred_last).abs()
        pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)

        pseudo_gt_fwd_flow_viz = flow_tensor_to_np(inp_dict['f0x'][0]) / 255.
        pseudo_gt_bwd_flow_viz = flow_tensor_to_np(inp_dict['f1x'][0]) / 255.
        fwd_flow_viz = flow_tensor_to_np(results_dict['f0x'][0]) / 255.
        bwd_flow_viz = flow_tensor_to_np(results_dict['f1x'][0]) / 255.
        viz_flow = torch.cat((torch.from_numpy(pseudo_gt_fwd_flow_viz).cuda(), torch.from_numpy(fwd_flow_viz).cuda(),
                              torch.from_numpy(bwd_flow_viz).cuda(), torch.from_numpy(pseudo_gt_bwd_flow_viz).cuda()),
                             dim=-1)
        return {
            'flow': viz_flow,
            'pred': pred_concat[0],
        }

    @staticmethod
    def normalize_w_rgb_mean(x0, x1, t):
        """
            return 0-centered x0, x1
        """
        t = t.unsqueeze(-1).unsqueeze(-1)

        x0, x1 = x0 / 255., x1 / 255.
        mean_ = torch.cat((x0, x1), dim=2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        x0, x1 = x0 - mean_, x1 - mean_
        return x0, x1, t, mean_