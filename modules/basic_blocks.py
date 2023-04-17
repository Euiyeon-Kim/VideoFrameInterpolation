import torch.nn as nn
from utils import initialize_weights


def conv_prelu(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_c)
    )


# Originated from https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020
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
        self.prelu = nn.PReLU(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


def make_residual_layers(nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(ResBlocknoBN(nf))
    return layers


# Originated from https://github.com/ltkong218/IFRNet
class HalfChannelConv5ResBlock(nn.Module):
    def __init__(self, in_c, side_c, bias=True):
        super(HalfChannelConv5ResBlock, self).__init__()
        self.side_c = side_c
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_c, side_c, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_c)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_c)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_c, side_c, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_c)
        )
        self.conv5 = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_c)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_c:, :, :] = self.conv2(out[:, -self.side_c:, :, :])
        out = self.conv3(out)
        out[:, -self.side_c:, :, :] = self.conv4(out[:, -self.side_c:, :, :])
        out = self.prelu(x + self.conv5(out))
        return out


class FeadForward(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU):
        super(FeadForward, self).__init__()
        out_features = out_dim or in_dim
        hidden_features = hidden_dim or in_dim
        self.fc1 = nn.Conv2d(in_dim, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        return out
