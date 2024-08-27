from torch import nn

from modules.block import ConvReluBN


class SecretDecoder(nn.Module):
    def __init__(self, cover_ch=3, secret_ch=4, hidden_ch=16):
        super(SecretDecoder, self).__init__()
        self.layers = nn.Sequential(
            ConvReluBN(cover_ch, hidden_ch, kernel_size=3, padding=1, stride=1),
            ConvReluBN(hidden_ch, hidden_ch, kernel_size=3, padding=1, stride=1),
            ConvReluBN(hidden_ch, hidden_ch, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(hidden_ch, secret_ch, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, stego):
        return self.layers(stego)
