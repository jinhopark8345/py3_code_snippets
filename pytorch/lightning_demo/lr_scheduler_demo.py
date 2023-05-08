
import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path
from torch.utils.data import Dataset

import numpy as np
import pytorch_lightning as pl
import torch

from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import math

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, LearningRateFinder
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ChainedScheduler
from torch.utils.data import DataLoader

from sconf import Config

@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")

class SimpleModelModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    def configure_optimizers(self):
        max_iter = None
        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)



class TestDataset(Dataset):
    def __init__(self, ds_size):
        super().__init__()
        # self.transform = ToTensor()
        self.ds_size = ds_size

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float), torch.tensor([idx], dtype=torch.float)
        # return torch.Tensor(idx), idx

    def __len__(self) -> int:
        return self.ds_size
def train(config):
    # define any number of nn.Modules (or use your current ones)
    # encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    # decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    model = nn.Sequential(nn.Linear(1, 1))

    # init the autoencoder
    model_module = SimpleModelModule(model, config)
    # autoencoder = LitAutoEncoder(encoder, decoder, config)
    # setup data
    # dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    dataset = TestDataset(config.num_training_samples_per_epoch)
    train_loader = utils.data.DataLoader(dataset)

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    # custom_ckpt = CustomCheckpointIO(config.ckpt_name)
    lr_callback = LearningRateMonitor(logging_interval="step")
    # lr_finder_callback = FineTuneLearningRateFinder(milestones=(config.lr_milestone))
    model_summary_callback = RichModelSummary(max_depth=config.model_summary_depth)

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_metric",
    #     dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
    #     filename=config.ckpt_name,
    #     save_top_k=config.save_top_k,
    #     save_last=False,
    #     mode="min",
    # )



    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        # gpus=torch.cuda.device_count(),
        # strategy="ddp_find_unused_parameters_true",
        strategy="ddp",
        accelerator="gpu",
        # plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        num_sanity_val_steps=config.num_sanity_val_steps,
        # precision=16,
        logger=logger,
        callbacks=[lr_callback, model_summary_callback],

    )

    trainer.fit(model=model_module, train_dataloaders=train_loader)

    # Step 5: use the model

    # # load checkpoint
    # checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    # autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # # choose your trained nn.Module
    # encoder = autoencoder.encoder
    # encoder.eval()

    # # embed 4 fake images!
    # fake_image_batch = Tensor(4, 28 * 28)
    # embeddings = encoder(fake_image_batch)
    # print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)


    # Step 6: Visualize Training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version
    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)

    train(config)
