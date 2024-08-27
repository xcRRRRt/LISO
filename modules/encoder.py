import torch
from torch import nn

from modules.block import ConvReluBN


class Encoder(nn.Module):
    def __init__(self, cover_in_ch=3, secret_in_ch=4, out_ch=16, gru_hidden_ch=16):
        """

        :param cover_in_ch: cover channel
        :param secret_in_ch: secret channel
        :param out_ch: cover feature channel
        :param gru_hidden_ch: gru hidden channel
        """
        super(Encoder, self).__init__()
        self.out_ch = out_ch
        self.gru_hidden_ch = gru_hidden_ch
        self.cover_features = ConvReluBN(cover_in_ch, out_ch + gru_hidden_ch, kernel_size=3, stride=1, padding=1)

        self.features = nn.Sequential(
            ConvReluBN(out_ch + gru_hidden_ch + secret_in_ch, out_ch + gru_hidden_ch, kernel_size=3, stride=1, padding=1),
            ConvReluBN(out_ch + gru_hidden_ch, out_ch + gru_hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_ch + gru_hidden_ch, out_ch + gru_hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, cover, secret):
        cover_features = self.cover_features(cover)
        features = self.features(torch.cat((cover_features, secret), dim=1))
        gru_hidden, cover_features = torch.split(features, [self.out_ch, self.gru_hidden_ch], dim=1)
        return gru_hidden, torch.relu(cover_features)


class GRUInputEncoder(nn.Module):
    def __init__(self, cover_in_ch=3, hidden_ch=32):
        """

        :param cover_in_ch: cover channel
        :param hidden_ch:
        """
        super(GRUInputEncoder, self).__init__()

        self.gradient_encoder = nn.Sequential(
            nn.Conv2d(cover_in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.perturbation_encoder = nn.Sequential(
            nn.Conv2d(cover_in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.grad_pert_combine = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch * 2 - 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, cover_features, gradient, perturbation):
        gradient_features = self.gradient_encoder(gradient)
        perturbation_features = self.perturbation_encoder(perturbation)
        grad_pert_combine_features = self.grad_pert_combine(torch.cat((gradient_features, perturbation_features), dim=1))
        gru_input = torch.cat((cover_features, grad_pert_combine_features, perturbation), dim=1)
        return gru_input
