from pdb import set_trace

import torch
import torch.nn as nn

class IntrinsicNet(nn.Module):
    def __init__(self):
        super(IntrinsicNet, self).__init__()

        self.convs = {}
        self.convs[0] = nn.Conv2d(3, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.intrinsic_conv = nn.Conv2d(256, 4, 1) # org is 5

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.intrinsic_conv(out)
        out = out.mean(3).mean(2)
        out = self.sigmoid(out)

        out = 1000.0 * out

        return out
