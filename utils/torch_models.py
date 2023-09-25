from torch import nn
import torch


class Conv1dBnRelu(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Conv1dBnRelu, self).__init__()
        self.conv3 = nn.Conv1d(in_chans, out_chans // 3, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv1d(in_chans, out_chans // 3, kernel_size=5, padding=2, stride=1)
        self.conv7 = nn.Conv1d(in_chans, out_chans // 3, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm1d(out_chans)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = torch.cat(
            [
                self.conv3(x),
                self.conv5(x),
                self.conv7(x),
            ],
            dim=1,
        )
        x = self.bn(x)
        x = self.relu(x)
        return x
