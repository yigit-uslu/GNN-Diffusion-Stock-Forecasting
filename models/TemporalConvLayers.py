import torch.nn as nn
import torch

class TemporalConvLayer(nn.Module):
    def __init__(self, conv, chop, act, norm, res_connection = None):
        super(TemporalConvLayer, self).__init__()
        # self.conv = conv
        # self.chop = chop
        # self.act = act
        # self.norm = norm
        self.conv = nn.Sequential(
            conv,
            chop,
            act,
            norm
        )
        self.res_connection = res_connection

    def forward(self, x):
        x_skip = self.res_connection(x) if self.res_connection is not None else x
        x = self.conv(x)

        x = x + x_skip if self.res_connection is not None else x
        return x