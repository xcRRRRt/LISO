from typing import Literal

from torch import nn


class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act: Literal['relu', 'leaky_relu'] = 'leaky_relu'):
        super(ConvReluBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if act == 'leaky_relu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif act == 'relu':
            self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x
