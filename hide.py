import warnings

import torch
import torchvision
from PIL import Image

from model import LISO, SECRET_TYPE
from util.data.datamodule import get_transformer

dir_ = "./data/test_image"


def inference(ckpt_path: str, cover_path: str, secret_path: str = None, save_path: str = "./result.jpg"):
    """
    LISO inference
    :param ckpt_path: <path/to/your/checkpoint>.ckpt
    :param cover_path: must be provide
    :param secret_path: must be provide if ``secret_type`` is ``image``
    :param save_path: result image save path, it will be cover[1,1], secret[1,2], stego[2,1], secret-recover[2,2] if ``secret type`` is ``image``, or will be cover[1,1], stego[2,1] if ``secret type`` is ``binary``
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LISO.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    hyper_parameters = torch.load(ckpt_path)["hyper_parameters"]
    secret_type: SECRET_TYPE = hyper_parameters["secret_type"]
    print("secret type: ", secret_type)
    if secret_type == "binary":
        if secret_type is not None:
            warnings.warn("Warning: `secret_path` is not necessary for `secret_type='binary'`", UserWarning)
    elif secret_type == "image":
        if secret_type is None:
            raise TypeError("Missing argument: `secret_path` is necessary for `secret_type='image'`")
    cover_size = hyper_parameters["cover_size"]
    secret_size = hyper_parameters["secret_size"]

    cover = Image.open(cover_path).convert("RGB")
    cover_transformer = get_transformer(cover_size[1:])
    cover = cover_transformer(cover).unsqueeze(0).to(device)
    secret = None
    if secret_type == "image":
        secret = Image.open(secret_path).convert("RGB")
        secret_transformer = get_transformer(secret_size[1:])
        secret = secret_transformer(secret).unsqueeze(0).to(device)
    elif secret_type == "binary":
        secret = torch.zeros(*secret_size).random_(0, 2).to(device)
    model_input = (cover, secret)

    model_output = model(model_input)
    stego, secret_recoveries = model_output
    grid = None
    if secret_type == "image":
        grid = torchvision.utils.make_grid(torch.cat((cover, secret, stego, secret_recoveries[-1]), dim=0), nrow=2)
        cover_psnr, cover_ssim, secret_psnr, secret_ssim = model.cal_metric(cover, stego, secret, secret_recoveries)
        print(f"cover_psnr: {cover_psnr.item()}, cover_ssim: {cover_ssim.item()}, secret_psnr: {secret_psnr.item()}, secret_ssim: {secret_ssim.item()}")
    elif secret_type == "binary":
        grid = torchvision.utils.make_grid(torch.cat((cover, stego), dim=0), nrow=1)
        cover_psnr, cover_ssim, acc = model.cal_metric(cover, stego, secret, secret_recoveries)
        print(f"cover_psnr: {cover_psnr.item()}, cover_ssim: {cover_ssim.item()}", f"acc: {acc.item()}")
    torchvision.utils.save_image(grid, save_path)


if __name__ == '__main__':
    inference("./lightning_logs/version_2/checkpoints/epoch=99-step=39999.ckpt", cover_path="./data/test_image/cover.jpg", secret_path="./data/test_image/secret.jpg")
