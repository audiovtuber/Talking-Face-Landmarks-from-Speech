import glob
from typing import Set

import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd


class GridDataset(torch.utils.data.Dataset):
    """PyTorch Dataset representing the [GRID corpus](https://spandh.dcs.shef.ac.uk//gridcorpus/)"""

    def _construct_data(self, individuals: Set[int]):
        lmark_paths = sorted(glob.glob("grid_dataset/features/*-frames.npy"))
        mel_paths = sorted(glob.glob("grid_dataset/features/*-melfeatures.npy"))
        data = {"melfeatures": mel_paths, "frames": lmark_paths}
        df = pd.DataFrame(data)

        df["video_id"] = df["frames"].str.extract(r"([a-zA-Z0-9]+)\.mpg")
        # mel_id is just used for validation of video_id
        mel_id = df["melfeatures"].str.extract(r"([a-zA-Z0-9]+)\.mpg")
        assert len(df) == sum(mel_id[0] == df["video_id"])

        df["individual"] = df["frames"].str.extract(r"-s([0-9]{1,2})-").astype(int)
        df = df[df["individual"].isin(individuals)]

        return df

    def __init__(
        self,
        individuals: Set[int],
        transform=None,
        target_transform=None,
        frame_delay: int = 0,
    ):
        super().__init__()
        self.df = self._construct_data(individuals)
        self.transform = transform
        self.target_transform = target_transform
        self.frame_delay = frame_delay

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mel = torch.tensor(
            np.load(open(self.df.iloc[idx]["melfeatures"], "rb"))
        ).float()
        frames = np.load(open(self.df.iloc[idx]["frames"], "rb"))

        # TODO: Consider not rolling from ends and just copy/ignore those edge cases
        if self.frame_delay != 0:
            frames = np.roll(frames, self.frame_delay, axis=0)
        frames = torch.tensor(frames).float()

        return (mel, frames.flatten(1))


class GridDataModule(pl.LightningDataModule):
    """[GRID corpus](https://spandh.dcs.shef.ac.uk//gridcorpus/) data module used primarily for training and validation"""

    def __init__(
        self,
        training_individuals: Set[int] = None,
        num_workers: int = 11,
        batch_size=32,
        frame_delay: int = 0,
    ):
        super().__init__()
        allowed_individuals = set(range(1, 35)) - {21}
        if training_individuals is None:
            training_individuals = set(range(1, 27)) - {21}
        assert 0 < len(training_individuals) < 33
        assert training_individuals.issubset(allowed_individuals)
        self.training_individuals = training_individuals
        self.val_individuals = allowed_individuals - self.training_individuals
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frame_delay = frame_delay

    def train_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def prepare_data(self):  # node-level data preparation. e.g. move files around
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GridDataset(
                individuals=self.training_individuals, frame_delay=self.frame_delay
            )
            self.val_dataset = GridDataset(
                individuals=self.val_individuals, frame_delay=self.frame_delay
            )


if __name__ == "__main__":
    data_module = GridDataModule(frame_delay=-2)
    data_module.setup("fit")
