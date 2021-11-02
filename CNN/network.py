import torch
import torch.nn as nn

"""
A small network version of DeepDTI.
The default channels in mid conv layers are 24, config it in train.py :model = DeepCNN(24)

It contained one in conv layer, one out conv layer, and 5 mid con layers 
and one skip connection for residual learning.

"""


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DeepCNN(nn.Module):

    def __init__(self, mid_channels):
        super(DeepCNN, self).__init__()
        self.inc = nn.Conv3d(in_channels=9, out_channels=mid_channels,
                             kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.conv1 = Conv(mid_channels, mid_channels)
        self.conv2 = Conv(mid_channels, mid_channels)
        self.conv3 = Conv(mid_channels, mid_channels)
        self.conv4 = Conv(mid_channels, mid_channels)
        self.conv5 = Conv(mid_channels, mid_channels)

        self.outc = nn.Conv3d(in_channels=mid_channels, out_channels=7,
                              kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        
        x1 = self.inc(x)
        x1 = self.relu(x1)

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)

        x1 = self.outc(x1) + x[:, :7, :, :, :]
        return x1
