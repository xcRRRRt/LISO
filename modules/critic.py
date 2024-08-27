import torch
from torch import nn

from modules.block import ConvReluBN


class Critic(nn.Module):
    def __init__(self, cover_in_ch, hidden_ch):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            ConvReluBN(cover_in_ch, hidden_ch, 3, 1, 1),
            ConvReluBN(hidden_ch, hidden_ch, 3, 1, 1),
            ConvReluBN(hidden_ch, hidden_ch, 3, 1, 1),
            nn.Conv2d(hidden_ch, 1, 3, 1, 1),
        )

    def forward(self, image):
        x = self.layers(image)
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        return x
