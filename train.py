import pytorch_lightning as pl

from model import LISO, SECRET_TYPE
from util.data.datamodule import LISODataModule

if __name__ == '__main__':
    hiding_type: SECRET_TYPE = "binary"
    cover_size = (3, 128, 128)
    secret_size = (3, 128, 128)
    iters = 15
    hidden_ch = 32
    eta = 1.0
    gamma = 0.8

    lr = 1e-4

    train_csv_path = "data/mini-imagenet-train.csv"
    val_csv_path = "data/mini-imagenet-valid.csv"

    batch_size = 2
    num_workers = 4

    data_module = LISODataModule(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        cover_size=cover_size,
        secret_size=secret_size,
        secret_type=hiding_type,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = LISO(
        cover_size=cover_size,
        secret_size=secret_size,
        secret_type=hiding_type,
        iters=iters,
        hidden_ch=hidden_ch,
        eta=eta,
        gamma=gamma,
        lr=lr,
    )
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_epochs=100
    )
    trainer.fit(model, data_module)
