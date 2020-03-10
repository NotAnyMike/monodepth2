import torch
import torch.nn as nn

class IntrinsicNet(nn.Module):
    def __init__(self, batch_size):
        super(IntrinsicNet, self).__init__()

        self.convs = {}
        self.convs[0] = nn.Conv2d(3, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.intrinsic_conv = nn.Conv2d(256, 5, 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        K = nn.Parameter(torch.eye(4).float(), requires_grad=False).view(1,4,4)
        self.K = K.repeat(batch_size, 1, 1)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.intrinsic_conv(out)
        out = out.mean(3).mean(2)

        #out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
        out = 0.01 * out.view(-1, 6)

        self.K[:, 0, 0] = out[:, 0]
        self.K[:, 1, 1] = out[:, 1]
        self.K[:, 0, 2] = out[:, 2]
        self.K[:, 1, 2] = out[:, 3]
        self.K[:, 0, 1] = out[:, 4]

        return self.K
