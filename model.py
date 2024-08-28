import gc
from collections.abc import Sequence
from typing import Literal

import pytorch_lightning as pl
import torchvision
from torch import nn
from torch.optim import Adam
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from modules.iterative_optimizer import IterativeOptimizer

SECRET_TYPE = Literal['image', 'binary']


class LISO(pl.LightningModule):
    def __init__(
            self,
            cover_size: Sequence[int],
            secret_size: Sequence[int],
            secret_type: SECRET_TYPE = 'binary',
            iters: int = 15,
            hidden_ch: int = 32,
            eta: float = 1.0,
            gamma: float = 0.8,
            lambda_: float = 1.0,
            miu: float = 1.0,
            lr: float = 1e-4,
    ):
        """
        LISO, `without critic`

        paper is here: https://openreview.net/pdf?id=gLPkzWjdhBN

        :param cover_size: cover size, must be C*H*W
        :param secret_size: secret size, must be C*H*W
        :param secret_type: must be ``binary`` or ``image``
        :param iters: iter times of iterative optimizer
        :param hidden_ch: the hidden channels that most of the blocks use
        :param eta: the ``η`` in ``algorithm1`` and ``figure2`` in the paper
        :param gamma: the ``γ`` in ``equation2``, it's a decay factor
        :param lambda_: the ``λ`` in ``equation2``, it's quality loss weight
        :param miu: the ``μ`` in ``equation2``, it denotes critic loss weight (`unused, critic is not implemented in this program`)
        :param lr: learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        cover_ch, secret_ch = cover_size[0], secret_size[0]
        assert secret_type in ["binary", "image"], "hiding_type must be either 'binary' or 'image'"
        assert len(cover_size) == 3, "cover_size must be of C*H*W"
        assert len(secret_size) == 3, "secret_size must be of C*H*W"
        self.lr = lr
        self.lambda_ = lambda_
        self.miu = miu
        self.secret_type = secret_type
        # self.critic = Critic(cover_ch, hidden_ch)
        self.iterative_optimizer = IterativeOptimizer(
            iters=iters,
            cover_in_ch=cover_ch,
            secret_in_ch=secret_ch,
            secret_type=secret_type,
            hidden_ch=hidden_ch,
            eta=eta
        )
        self.cover_encoder = self.iterative_optimizer.cover_encoder
        self.secret_decoder = self.iterative_optimizer.secret_decoder

        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

        self.cover_loss = nn.MSELoss()
        if secret_type == 'binary':
            self.secret_loss = nn.BCEWithLogitsLoss()
        elif secret_type == 'image':
            self.secret_loss = nn.MSELoss()

        self.weights = list(reversed([gamma ** x for x in range(iters)]))

    def cal_acc(self, secret, secret_recovery):
        acc = (secret_recovery >= 0).eq(secret >= 0.5).sum().float() / secret.numel()
        return acc

    def cal_metric(self, cover, stego, secret, secret_recoveries):
        psnr = self.psnr(cover, stego)
        ssim = self.ssim(cover, stego)
        if self.secret_type == "binary":
            accs = [self.cal_acc(secret, sr) for sr in secret_recoveries]
            return psnr, ssim, max(accs)
        elif self.secret_type == "image":
            secret_psnr = self.psnr(secret, secret_recoveries[-1])
            secret_ssim = self.ssim(secret, secret_recoveries[-1])
            return psnr, ssim, secret_psnr, secret_ssim

    def _cal_loss(self, secret, secret_recoveries, cover, stegos, stage: str):
        loss = 0
        for weight, secret_recovery, stego in zip(self.weights, secret_recoveries, stegos):
            secret_l = self.secret_loss(secret_recovery, secret)
            cover_l = self.cover_loss(cover, stego)
            loss += (secret_l + cover_l * self.lambda_) * weight
        self.log(f"{stage}/loss", loss)
        return loss

    def step(self, batch, batch_idx, stage: str):
        cover, secret = batch
        stegos = self.iterative_optimizer(cover, secret)
        secret_recoveries = [self.secret_decoder(stego) for stego in stegos]
        gc.collect()
        loss = self._cal_loss(secret, secret_recoveries, cover, stegos, stage)
        cover_psnr, cover_ssim, *secret_metric = self.cal_metric(cover, stegos[-1], secret, secret_recoveries)
        self.log(f"{stage}/cover_psnr", cover_psnr)
        self.log(f"{stage}/cover_ssim", cover_ssim)
        if self.secret_type == "binary":
            acc = secret_metric[0]
            self.log(f"{stage}/cover_acc", acc)
        elif self.secret_type == "image":
            secret_psnr, secret_ssim = secret_metric
            self.log(f"{stage}/secret_psnr", secret_psnr)
            self.log(f"{stage}/secret_ssim", secret_ssim)
        if stage == "val" and batch_idx == 0:
            self.logger.experiment.add_image("img/cover", torchvision.utils.make_grid(cover), self.global_step)
            self.logger.experiment.add_image("img/stego", torchvision.utils.make_grid(stegos[-1]), self.global_step)
            if self.secret_type == "image":
                self.logger.experiment.add_image("img/secret", torchvision.utils.make_grid(secret), self.global_step)
                self.logger.experiment.add_image("img/secret_recovery", torchvision.utils.make_grid(secret_recoveries[-1]), self.global_step)

        return loss, stegos, secret_recoveries

    def training_step(self, batch, batch_idx):
        loss, stegos, secret_recoveries = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'val')

    def forward(self, x):
        cover, secret = x
        stegos = self.iterative_optimizer(cover, secret)
        secret_recoveries = [self.secret_decoder(stego) for stego in stegos]
        return stegos[-1], secret_recoveries

    def configure_optimizers(self):
        params = list(self.iterative_optimizer.parameters())
        # critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
        coder_optimizer = Adam(params, lr=self.lr)
        return coder_optimizer
