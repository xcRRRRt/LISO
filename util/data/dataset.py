import random
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


def _get_image(img_path: str, transform: Compose) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    return img


class BaseDataset(Dataset):
    def __init__(self, csv_path: str, transform: Compose | ToTensor, limit: int = 10000):
        super(BaseDataset, self).__init__()
        self.img_paths: list[str] = pd.read_csv(csv_path)['path'].tolist()
        random.shuffle(self.img_paths)
        self.limit = limit
        self.transform = transform

    def __len__(self):
        return min(self.limit, len(self.img_paths))

    def __getitem__(self, idx):
        return _get_image(self.img_paths[idx], self.transform)


class BinaryHidingDataset(BaseDataset):
    def __init__(self, csv_path: str, transform: Compose | ToTensor, secret_shape: Sequence[int] = (3, 128, 128), limit: int = 10000):
        super(BinaryHidingDataset, self).__init__(csv_path, transform, limit)
        self.secret_shape = secret_shape

    def __getitem__(self, idx):
        img = _get_image(self.img_paths[idx], self.transform)
        secret = torch.zeros(*self.secret_shape).random_(0, 2)
        return img, secret


class ImageHidingDataset(BaseDataset):
    def __init__(self, csv_path: str, cover_transform: Compose | ToTensor, secret_transform: Compose | ToTensor, limit: int = 10000):
        super(ImageHidingDataset, self).__init__(csv_path, cover_transform, limit)
        self.secret_transform = secret_transform
        self._gen_image_pairs()

    def __getitem__(self, idx):
        cover_path, secret_path = self.image_pairs[idx]
        cover = _get_image(cover_path, self.transform)
        secret = _get_image(secret_path, self.secret_transform)
        return cover, secret

    def __len__(self):
        return min(self.max_pairs, self.limit)

    def _gen_image_pairs(self) -> None:
        image_pairs: set[tuple[str, str]] = set()
        self.max_pairs = len(self.img_paths) * len(self.img_paths) - 1
        while len(image_pairs) < min(self.max_pairs, self.limit):
            pair: tuple[str, str] = tuple(random.sample(self.img_paths, 2))  # type: ignore
            image_pairs.add(pair)
        self.image_pairs: list[tuple[str, str]] = list(image_pairs)


if __name__ == '__main__':
    import torchvision

    train_csv_path = r"E:\data-hiding\code_paper\benchmark\LISO\data\mini-imagenet-train.csv"
    print("Binary Hiding Dataset")
    bhd = BinaryHidingDataset(train_csv_path, transform=torchvision.transforms.ToTensor())
    print(len(bhd))
    cover, secret = next(iter(bhd))
    print(cover.shape, secret.shape)
    print(cover)

    # print("Image Hiding Dataset")
    # ihd = ImageHidingDataset(train_csv_path, cover_transform=torchvision.transforms.ToTensor(), secret_transform=torchvision.transforms.ToTensor())
    # print(len(ihd))
    # cover, secret = next(iter(ihd))
    # print(cover.shape, secret.shape)
