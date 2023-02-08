import torch

from utils.flow_viz import flow_tensor_to_np


class Logger:
    def __init__(self, summary_writer, metric_summary_freq=100, start_step=0):
        self.summary_writer = summary_writer

        self.total_steps = start_step
        self.metric_summary_freq = metric_summary_freq

        self.running_loss = {}

    def print_training_status(self, mode='train'):
        print(f'Step: {self.total_steps:06d} \t total: {(self.running_loss["total_loss"] / self.metric_summary_freq):.3f}')
        for k in self.running_loss:
            self.summary_writer.add_scalar(mode + '/' + k,
                                           self.running_loss[k] / self.metric_summary_freq, self.total_steps)
            self.running_loss[k] = 0.0
        self.summary_writer.flush()

    def push(self, metrics, mode='train'):
        self.total_steps += 1

        # Running mean
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]

        # Loggin on tensorboard
        if self.total_steps % self.metric_summary_freq == 0:
            self.print_training_status(mode)
            self.running_loss = {}

    def write_dict(self, results, step=None):
        log_step = step if step is not None else self.total_steps
        for key in results:
            tag = key.split('_')[0]
            tag = tag + '/' + key
            self.summary_writer.add_scalar(tag, results[key], log_step)
        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()

    def add_image_summary(self, img_dict):
        for k, v in img_dict.items():
            self.summary_writer.add_image(k, v, self.total_steps)
        self.summary_writer.flush()

    # def add_image_summary(self, x0, x1, xt, results_dict):
    #     x0_01, x1_01, xt_01 = x0[0] / 255., x1[0]/255., xt[0] / 255.
    #     pred_last = results_dict['frame_preds'][-1][0][None]
    #
    #     fwd_flow_viz = flow_tensor_to_np(results_dict['f01'][0]) / 255.
    #     bwd_flow_viz = flow_tensor_to_np(results_dict['f10'][0]) / 255.
    #     viz_flow = torch.cat((x0_01, torch.from_numpy(fwd_flow_viz).cuda(),
    #                           torch.from_numpy(bwd_flow_viz).cuda(), x1_01), dim=-1)
    #     self.summary_writer.add_image('flow', viz_flow, self.total_steps)
    #
    #     xt_warp_x0_01 = results_dict['xt_warp_x0'][0]
    #     xt_warp_x1_01 = results_dict['xt_warp_x1'][0]
    #     x0_mask = results_dict['x0_mask'][0].repeat(3, 1, 1).unsqueeze(0)
    #     process_concat = torch.cat((xt_warp_x0_01[None], x0_mask, xt_warp_x1_01[None]), dim=-1)
    #     self.summary_writer.add_image('process', process_concat[0], self.total_steps)
    #
    #     half = (x0_01 + x1_01) / 2
    #     err_map = (xt_01 - pred_last).abs()
    #     pred_concat = torch.cat((half[None], pred_last, xt_01[None], err_map), dim=-1)
    #     self.summary_writer.add_image('pred', pred_concat[0], self.total_steps)
    #     self.summary_writer.flush()
