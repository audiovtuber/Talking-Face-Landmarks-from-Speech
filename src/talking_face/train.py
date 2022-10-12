import glob
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple, Sequence, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import PyTorchProfiler
from torch.optim import Adam, SGD
import wandb

from talking_face.dataset import GridDataModule


class TalkingFaceLSTM(pl.LightningModule):
    """
    The main workhorse for training and predicting face landmarks from audio
    """

    def __init__(
        self,
        num_landmarks,
        optimizer="adam",
        lr=1e-3,
        layers=4,
        hidden_size=256,
    ):
        super().__init__()

        optimizers = {"adam": Adam, "sgd": SGD}
        self.optimizer = optimizers[optimizer]
        self.num_landmarks = num_landmarks
        self.lr = lr
        self.layers = layers
        self.hidden_size = hidden_size
        # instantiate loss criterion
        self.criterion = nn.MSELoss()
        self.model = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            proj_size=2 * self.num_landmarks,
            num_layers=self.layers,
            dropout=0.2,
            batch_first=True,
        )
        # self.train_accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()

    @torch.jit.ignore
    def forward(self, X, hidden=None, cell=None):
        if hidden is not None:
            return self.model(X, (hidden, cell))
        else:
            return self.model(X)

    @torch.jit.export
    def predict(
        self, batch, hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        out, hiddens = self.model(batch, hiddens)
        return out.reshape((-1, 68, 2)), hiddens

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # drops the hidden state because it's not needed during batch operations, only during single-frame inference
        preds = self(x)[0]
        loss = self.criterion(preds, y)
        # perform logging
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # drops the hidden state because it's not needed during batch operations, only during single-frame inference
        preds = self(x)[0]
        loss = self.criterion(preds, y)
        # perform logging
        self.log("val_loss", loss, prog_bar=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--landmarks",
        type=int,
        default=68,
        help="Number of landmarks to predict during training",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        help="Type of optimizer to use. Adam or SGD",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Starting learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Samples per batch",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Number of units per LSTM layer. Must be >= 2 * landmarks",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total epochs for training",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=250,
        help="Log to Weights & Biases after N steps",
    )
    parser.add_argument(
        "--frame-delay",
        type=int,
        default=0,
        help="Offsets the landmarks by this many frames (1 = 40ms, 2 = 80ms, etc)",
    )
    parser.add_argument(
        "--profile",
        required=False,
        action="store_true",
        help="Runs the PyTorch Profiler",
    )
    return parser.parse_args()


def export_to_torchscript(model: Union[TalkingFaceLSTM, str], output_dir: str = None):
    model = (
        TalkingFaceLSTM.load_from_checkpoint(model) if isinstance(model, str) else model
    )
    torchscript_model = model.to_torchscript()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        target_filepath = Path(output_dir) / "torchscript_model.pt"
        torch.jit.save(torchscript_model, target_filepath)
        print(f"Saved as {target_filepath}")

    return torchscript_model


# NOTE: TalkingFaceLSTM does not support ONNX at this time due to use of projections
def export_to_onnx(model: str, input_dims: Sequence[int], output_dir):
    """
    model = TalkingFaceLSTM.load_from_checkpoint(model)
    os.makedirs(output_dir, exist_ok=True)
    target_filepath = Path(output_dir) / "model.onnx"
    model.to_onnx(target_filepath, input_sample=torch.randn(input_dims), export_params=True)
    """
    raise NotImplementedError(
        "TalkingFaceLSTM does not support ONNX at this time due to use of projections"
    )


if __name__ == "__main__":
    args = parse_args()
    args.optimizer = args.optimizer.lower()

    # TODO: Someday, maybe support other number of landmarks
    assert args.landmarks == 68
    assert args.optimizer in {"adam", "sgd"}
    assert args.hidden_size >= 2 * args.landmarks
    """
    args = SimpleNamespace(
        num_landmarks=136,
        optimizer='adam',
        learning_rate=1e-3,
        batch_size=128,
        # Trainer args
        gpus=int(torch.cuda.is_available()),
        epochs=50,
    )
    """
    # os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_DISABLE_CODE"] = "True"
    # # Instantiate Model
    model = TalkingFaceLSTM(
        num_landmarks=args.landmarks,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        layers=args.layers,
        hidden_size=args.hidden_size,
    )
    # Instantiate lightning trainer and train model
    wandb_logger = WandbLogger(project="audio-vtuber", job_type="train", log_model=True)
    wandb_logger.watch(model, log_freq=max(100, args.log_every_n_steps))
    wandb_logger.log_hyperparams(vars(args))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer_args = {
        "gpus": int(torch.cuda.is_available()),
        "max_epochs": args.epochs,
    }
    if args.profile:
        profiler = PyTorchProfiler(
            export_to_chrome=True, dirpath=wandb_logger.experiment.dir
        )
        trainer_args["profiler"] = profiler
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **trainer_args,
    )

    data_module = GridDataModule(
        batch_size=args.batch_size, frame_delay=args.frame_delay
    )
    trainer.fit(
        model,
        datamodule=data_module,
    )

    if args.profile:
        try:  # add execution trace to logged and versioned binaries
            trace_matcher = wandb_logger.experiment.dir + "/*.pt.trace.json"
            trace_files = glob.glob(trace_matcher)
            trace_at = wandb.Artifact(
                name=f"trace-{wandb_logger.experiment.id}", type="trace"
            )
            for trace_file in trace_files:
                trace_at.add_file(trace_file, name=Path(trace_file).name)
            wandb.log_artifact(trace_at)
        except IndexError:
            print("trace not found")
    """
    NOTE: Three hacky things here:
    1. wandb seems to monkeypatch the model, so if we try to directly export it to torchscript, it may fail.
        As a workaround, we simply reload the model before exporting (and subsequently uploading)
    2. We hijack the checkpoint_callback to save the exported the torchscript model to the same directory
        TODO: Write a custom ModelCheckpoint() that does this for us instead
    3. Really, the staged model should be tagged and compared to see if it's better than whatever exists already (so it can be considered latest and auto-pulled by deployed services)
    """
    model = TalkingFaceLSTM.load_from_checkpoint(checkpoint_callback.best_model_path)
    torchscript_model = export_to_torchscript(
        model, output_dir=checkpoint_callback.dirpath
    )
    staged_model_artifact = wandb.Artifact(
        f"torchscript_model-{wandb_logger.experiment.id}", type="prod-ready"
    )
    staged_model_artifact.add_file(
        Path(checkpoint_callback.dirpath) / "torchscript_model.pt"
    )
    wandb.log_artifact(staged_model_artifact)
