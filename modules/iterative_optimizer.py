from typing import Literal

import torch
from torch import nn

from modules.decoder import SecretDecoder
from modules.encoder import GRUInputEncoder, Encoder


class GRU(nn.Module):
    def __init__(self, input_ch=80, gru_hidden_ch=16):
        super(GRU, self).__init__()
        self.convz = nn.Conv2d(gru_hidden_ch + input_ch, gru_hidden_ch, 3, padding=1)
        self.convr = nn.Conv2d(gru_hidden_ch + input_ch, gru_hidden_ch, 3, padding=1)
        self.convq = nn.Conv2d(gru_hidden_ch + input_ch, gru_hidden_ch, 3, padding=1)

    def forward(self, hidden, gru_in):
        hx = torch.cat([hidden, gru_in], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat((r * hidden, gru_in), dim=1)))
        hidden = (1 - z) * hidden + z * q
        return hidden


class IterativeOptimizerBlock(nn.Module):
    def __init__(self, cover_in_ch=3, gru_in_feature=80, gru_hidden_ch=16, hidden_size=32, eta=1.0):
        super(IterativeOptimizerBlock, self).__init__()
        self.eta = eta
        self.gru_in = GRUInputEncoder(cover_in_ch, hidden_ch=hidden_size)
        self.gru = GRU(input_ch=gru_in_feature, gru_hidden_ch=gru_hidden_ch)
        self.g_t = nn.Sequential(
            nn.Conv2d(gru_hidden_ch, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, cover_in_ch, kernel_size=3, padding=1),
        )

    def forward(self, cover_feature, gradient, perturbation, gru_hidden):
        gru_in = self.gru_in(cover_feature, gradient, perturbation)
        gru_hidden = self.gru(gru_hidden, gru_in)
        g_t = self.g_t(gru_hidden)
        perturbation = perturbation + self.eta * g_t
        return perturbation, gru_hidden


class IterativeOptimizer(nn.Module):
    def __init__(
            self,
            iters=15,
            cover_in_ch=3,
            secret_type: Literal['image', 'binary'] = 'binary',
            secret_in_ch=4,
            hidden_ch=32,
            eta=1.0
    ):
        super(IterativeOptimizer, self).__init__()
        cover_features = gru_hidden_ch = hidden_ch // 2
        gru_in_features = hidden_ch + hidden_ch + cover_features

        if secret_type == 'binary':
            self.secret_loss = nn.BCEWithLogitsLoss(reduction='sum')
        elif secret_type == 'image':
            self.secret_loss = nn.MSELoss(reduction='sum')

        self.iters = iters
        self.cover_encoder = Encoder(cover_in_ch, secret_in_ch, out_ch=cover_features, gru_hidden_ch=gru_hidden_ch)
        self.opt_block = IterativeOptimizerBlock(cover_in_ch=cover_in_ch, gru_in_feature=gru_in_features, gru_hidden_ch=gru_hidden_ch,
                                                 hidden_size=hidden_ch, eta=eta)
        self.secret_decoder = SecretDecoder(cover_in_ch, secret_in_ch, hidden_ch=hidden_ch)

    def cal_iter_grad(self, stego, secret):
        with torch.enable_grad():
            stego.requires_grad = True
            secret_recovery = self.secret_decoder(stego)
            loss = self.secret_loss(secret_recovery, secret)
            loss.backward()
            grad = stego.grad.detach()
            stego.requires_grad = False
        return grad

    def forward(self, cover, secret):
        stegos = []
        perturbation = cover.clone()
        cover_feature, gru_hidden_feature = self.cover_encoder(cover, secret)
        for _ in range(self.iters):
            perturbation = perturbation.detach()
            gradient = self.cal_iter_grad(perturbation, secret)
            perturbation, gru_hidden_feature = self.opt_block(cover_feature, gradient, perturbation, gru_hidden_feature)
            stegos.append(perturbation)
        return stegos
