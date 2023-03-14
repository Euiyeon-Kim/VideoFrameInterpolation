import torch.nn as nn
import torch.nn.functional as F

from utils import initialize_weights


class ResBlocknoBN(nn.Module):
    '''
        Residual block w/o BN
        ---Conv-ReLU-Conv-+-
         |________________|
    '''

    def __init__(self, nf=64):
        super(ResBlocknoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def make_layer(nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(ResBlocknoBN(nf))
    return nn.Sequential(*layers)