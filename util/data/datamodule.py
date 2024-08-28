from collections.abc import Sequence
from functools import partial
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SECRET_TYPE
from util.data.dataset import ImageHidingDataset, BinaryHidingDataset


def get_transformer(size_: Sequence[int]):
    return transforms.Compose([
        transforms.Resize(size_),
        transforms.ToTensor(),
    ])


class LISODataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_csv_path: str,
            val_csv_path: str,
            # test_csv_path: str = None,
            train_limit: int = 800,
            val_limit: int = 200,
            # test_limit: int = 200,
            secret_type: SECRET_TYPE = "binary",
            cover_size: Sequence[int] = (3, 128, 128),
            secret_size: Sequence[int] = (4, 128, 128),
            batch_size: int = 2,
            num_workers: int = 4,

    ):
        assert secret_type in ["binary", "image"], "hiding_type must be either 'binary' or 'image'"
        assert len(cover_size) == 3, "cover_size must be of C*H*W"
        assert len(secret_size) == 3, "secret_size must be of C*H*W"
        self.val_dataset = None
        self.train_dataset = None
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        # self.test_csv_path = test_csv_path
        self.train_limit = train_limit
        self.val_limit = val_limit
        # self.test_limit = test_limit
        self.secret_type = secret_type
        self.cover_size = cover_size
        self.secret_size = secret_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.secret_type == "binary":
            self.dataset_partial = partial(
                BinaryHidingDataset,
                transform=get_transformer(self.cover_size[1:]),
                secret_shape=self.secret_size,
            )
        elif self.secret_type == "image":
            self.dataset_partial = partial(
                ImageHidingDataset,
                cover_transform=get_transformer(self.cover_size[1:]),
                secret_transform=get_transformer(self.secret_size[1:]),
            )

        self.dataloader_partial = partial(
            DataLoader,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=True
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = self.dataset_partial(self.train_csv_path, limit=self.train_limit)
            self.val_dataset = self.dataset_partial(self.val_csv_path, limit=self.val_limit)

    def train_dataloader(self):
        return self.dataloader_partial(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.dataloader_partial(self.val_dataset, shuffle=False)
