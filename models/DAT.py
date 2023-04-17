import torch
import torch.nn as nn

from .BaseModel import BaseModel
import modules.losses as losses
from utils.flow_viz import flow_tensor_to_np
from modules.cnn_encoders import SameChannelResEncoder
from modules.query_builder import DCNInterFeatBuilderwithT
from modules.deformable_attn import CrossDeformableAttentionBlockwFlow
from modules.generator import BasicResPixelShuffleGenerator


class DATwConstantnCv1(BaseModel):
    def __init__(self, args):
        super(DATwConstantnCv1, self).__init__(args)
        self.nf = args.nf
        self.feature_encoder = SameChannelResEncoder(args.nf, args.enc_res_blocks)

        self.coarse_query_builder = DCNInterFeatBuilderwithT(args.nf)
        self.lv4_to_lv3 = nn.ConvTranspose2d(args.nf + 4, args.nf + 4, 4, 2, 1)

        self.dat_lv3 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=8, n_groups=4, n_heads=4,
                                                          offset_scale=2.0, mlp_ratio=args.mlp_ratio)

        self.lv3_to_lv2 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.dat_lv2 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=16, n_groups=8, n_heads=8,
                                                          offset_scale=4.0, mlp_ratio=args.mlp_ratio)

        self.lv2_to_lv1 = nn.ConvTranspose2d(args.nf, args.nf, 4, 2, 1)
        self.dat_lv1 = CrossDeformableAttentionBlockwFlow(args.nf, args.nf, n_samples=32, n_groups=8, n_heads=8,
                                                          offset_scale=8.0, mlp_ratio=args.mlp_ratio,
                                                          pred_res_flow=False)
        self.pixel_geneartor = BasicResPixelShuffleGenerator(args.nf, args.dec_res_blocks)

        self.distill_lambda = args.distill_lambda
        self.l1_loss = losses.Charbonnier_L1()
        self.tr_loss = losses.Ternary(7)
        self.rb_loss = losses.Charbonnier_Ada()

    def get_img_dict(self, *args, **kwargs):
        x0, x1, xt = self.img_dict['x0'], self.img_dict['x1'], self.img_dict['xt']
        half = (x0 + x1) / 2
        err_map = (xt - self.img_dict['pred']).abs()
        pred = torch.cat((half[None], self.img_dict['pred'][None], xt[None], err_map[None]), dim=-1)

        ft0_1 = self.img_dict['pred_ft0'][0]
        ft0_2 = self.img_dict['pred_ft0'][1]
        ft0_3 = self.img_dict['pred_ft0'][2]
        ft0_4 = self.img_dict['pred_ft0'][3]
        ft1_1 = self.img_dict['pred_ft1'][0]
        ft1_2 = self.img_dict['pred_ft1'][1]
        ft1_3 = self.img_dict['pred_ft1'][2]
        ft1_4 = self.img_dict['pred_ft1'][3]
        ft0_1_viz = torch.from_numpy(flow_tensor_to_np(ft0_1) / 255.).cuda()
        ft0_2_viz = torch.from_numpy(flow_tensor_to_np(ft0_2) / 255.).cuda()
        ft0_3_viz = torch.from_numpy(flow_tensor_to_np(ft0_3) / 255.).cuda()
        ft0_4_viz = torch.from_numpy(flow_tensor_to_np(ft0_4) / 255.).cuda()
        ft1_1_viz = torch.from_numpy(flow_tensor_to_np(ft1_1) / 255.).cuda()
        ft1_2_viz = torch.from_numpy(flow_tensor_to_np(ft1_2) / 255.).cuda()
        ft1_3_viz = torch.from_numpy(flow_tensor_to_np(ft1_3) / 255.).cuda()
        ft1_4_viz = torch.from_numpy(flow_tensor_to_np(ft1_4) / 255.).cuda()

        ft0_pseudo_gt = torch.from_numpy(flow_tensor_to_np(self.img_dict['ft0']) / 255.).cuda()
        ft1_pseudo_gt = torch.from_numpy(flow_tensor_to_np(self.img_dict['ft1']) / 255.).cuda()

        viz_flow = torch.cat((ft0_4_viz, ft0_3_viz, ft0_2_viz, ft0_1_viz,
                              ft0_pseudo_gt, ft1_pseudo_gt,
                              ft1_1_viz, ft1_2_viz, ft1_3_viz, ft1_4_viz), dim=-1)
        return {
            'pred': pred[0],
            'flow': viz_flow,
        }

    def _generate_frame(self, x0, x1, t, set_img_dict=False):
        x0, x1, mean_ = self.norm_w_rgb_mean(x0, x1)
        feat0_1, feat0_2, feat0_3, feat0_4 = self.feature_encoder(x0)
        feat1_1, feat1_2, feat1_3, feat1_4 = self.feature_encoder(x1)
        pred_feat_t_4, pred_ft0_4, pred_ft1_4 = self.coarse_query_builder(feat0_4, feat1_4, t)

        pred_scale_3 = self.lv4_to_lv3(torch.cat((pred_feat_t_4, pred_ft0_4, pred_ft1_4), dim=1))
        pred_feat_t_3 = pred_scale_3[:, :self.nf, :, :]
        pred_ft0_3, pred_ft1_3 = pred_scale_3[:, self.nf:self.nf+2, :, :],  pred_scale_3[:, self.nf+2:self.nf+4, :, :]

        attended_feat_t_3, pred_ft0_2, pred_ft1_2 = self.dat_lv3(pred_feat_t_3, feat0_3, feat1_3, pred_ft0_3, pred_ft1_3)

        query_feat_t_2 = self.lv3_to_lv2(attended_feat_t_3)
        attended_feat_t_2, pred_ft0_1, pred_ft1_1 = self.dat_lv2(query_feat_t_2, feat0_2, feat1_2, pred_ft0_2, pred_ft1_2)

        query_feat_t_1 = self.lv2_to_lv1(attended_feat_t_2)
        attended_feat_t_1 = self.dat_lv1(query_feat_t_1, feat0_1, feat1_1, pred_ft0_1, pred_ft1_1)
        img_pred = self.pixel_geneartor(attended_feat_t_1, mean_)

        if not self.training:
            return img_pred

        _intermediate_for_loss = {
            'pred_ft0': [
                self.resize(pred_ft0_1, 2.0), self.resize(pred_ft0_2, 4.0),
                self.resize(pred_ft1_3, 8.0), self.resize(pred_ft0_4, 16.0)
            ],
            'pred_ft1': [
                self.resize(pred_ft1_1, 2.0), self.resize(pred_ft1_2, 4.0),
                self.resize(pred_ft1_3, 8.0), self.resize(pred_ft1_4, 16.0)
            ]
        }
        if set_img_dict:
            self.img_dict = {
                'x0': x0[0] + mean_[0],
                'x1': x1[0] + mean_[0],
                'pred': img_pred[0],
                'pred_ft0': [_intermediate_for_loss['pred_ft0'][0][0], _intermediate_for_loss['pred_ft0'][1][0],
                             _intermediate_for_loss['pred_ft0'][2][0], _intermediate_for_loss['pred_ft0'][3][0]],
                'pred_ft1': [_intermediate_for_loss['pred_ft1'][0][0], _intermediate_for_loss['pred_ft1'][1][0],
                             _intermediate_for_loss['pred_ft1'][2][0], _intermediate_for_loss['pred_ft1'][3][0]],
            }
        return img_pred, _intermediate_for_loss

    def inference(self, x0, x1, t):
        return self._generate_frame(x0, x1, t, False)

    def forward(self, inp_dict, set_img_dict=False):
        x0, x1, xt, t = inp_dict['x0'], inp_dict['x1'], inp_dict['xt'], inp_dict['t']
        img_pred, _intermediate_outputs = self._generate_frame(x0, x1, t, set_img_dict)
        ft0, ft1 = inp_dict['f0x'], inp_dict['f1x']

        if set_img_dict:
            self.img_dict['xt'] = xt[0]
            self.img_dict['ft0'] = ft0[0]
            self.img_dict['ft1'] = ft1[0]

        # Calculate loss
        l1_loss = self.l1_loss(img_pred - xt)
        census_loss = self.tr_loss(img_pred, xt)
        total_loss = l1_loss + census_loss
        log_dict = {
            'l1_loss': l1_loss.item(),
            'census_loss': census_loss.item(),
        }

        if self.distill_lambda is not None:
            pred_ft0s, pred_ft1s = _intermediate_outputs['pred_ft0'], _intermediate_outputs['pred_ft1']

            robust_weight0 = losses.get_robust_weight(pred_ft0s[0], ft0, beta=0.3)
            robust_weight1 = losses.get_robust_weight(pred_ft1s[0], ft1, beta=0.3)
            distill_loss = self.distill_lambda * (self.rb_loss(pred_ft0s[1] - ft0, weight=robust_weight0) + \
                                                  self.rb_loss(pred_ft1s[1] - ft1, weight=robust_weight1) + \
                                                  self.rb_loss(pred_ft0s[2] - ft0, weight=robust_weight0) + \
                                                  self.rb_loss(pred_ft1s[2] - ft1, weight=robust_weight1) + \
                                                  self.rb_loss(pred_ft0s[3] - ft0, weight=robust_weight0) + \
                                                  self.rb_loss(pred_ft1s[3] - ft1, weight=robust_weight1))
            total_loss = total_loss + distill_loss
            log_dict.update({'flow_loss': distill_loss.item()})

        log_dict.update({'total_loss': total_loss.item()})
        return total_loss, log_dict
