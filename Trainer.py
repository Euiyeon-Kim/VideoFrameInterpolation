import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from evaluate import validate_vimeo90k, validate_ucf101

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args, local_rank, training=True):
        self.args = args

        # Model definition
        self.model_without_ddp = getattr(models, args.model_name)(args).to(DEVICE)

        if local_rank != -1:
            self.model = DDP(self.model_without_ddp, device_ids=[local_rank], output_device=local_rank)
        else:
            self.model = self.model_without_ddp

        if training:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)

    def load_trained(self, chkpt):
        self.model.load_state_dict(chkpt['model'])
        if 'optim' in chkpt.keys():
            self.optimizer.load_state_dict(chkpt['optim'])

    def get_img_dict(self):
        return self.model_without_ddp.get_img_dict()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def device(self):
        self.model.to(DEVICE)

    def save_model(self, path, epoch, step, best_psnr, save_optim=True):
        chkpt = {
            'model': self.model.state_dict(),
            'best_psnr': best_psnr,
            'step': step,
            'epoch': epoch,
        }
        if save_optim:
            chkpt.update({
                'optimizer': self.optimizer.state_dict(),
            })
        torch.save(chkpt, path)

    def inference(self, x0, x1, t):
        return self.model.inference(x0, x1, t)

    @torch.no_grad()
    def validate_vimeo90k(self, report_ssim=False):
        return validate_vimeo90k(self.args, self.model_without_ddp, report_ssim=report_ssim)

    @torch.no_grad()
    def validate_ucf101(self, report_ssim=False):
        return validate_ucf101(self.model_without_ddp, report_ssim=report_ssim)

    def one_step(self, inp_dict, lr, set_img_dict=False):
        # set lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        total_loss, log_dict = self.model(inp_dict, set_img_dict=set_img_dict)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return log_dict
