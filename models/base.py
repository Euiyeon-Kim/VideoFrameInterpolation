import torch
import torch.nn as nn


class Basemodel(nn.Module):
    def __init__(self, args):
        super(Basemodel, self).__init__()
        self.args = args

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