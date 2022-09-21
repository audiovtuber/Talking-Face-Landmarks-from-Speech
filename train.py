import os
from argparse import ArgumentParser
# from types import SimpleNamespace

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam, SGD
import torch.nn as nn

from dataset import GridDataModule


class TalkingFaceLSTM(pl.LightningModule):
    def __init__(self, num_landmarks,
                optimizer='adam', 
                lr=1e-3,
                layers=4,
                hidden_size=256,
                ):
        super().__init__()

        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        self.num_landmarks = num_landmarks
        self.lr = lr
        self.layers=layers
        self.hidden_size = hidden_size
        #instantiate loss criterion
        self.criterion = nn.MSELoss()
        self.model = nn.LSTM(input_size=128, hidden_size=self.hidden_size, proj_size=2 * self.num_landmarks, num_layers=self.layers, dropout=0.2, batch_first=True, )
        # self.train_accuracy = torchmetrics.Accuracy()

    def forward(self, X, hidden=None, cell=None):
        if hidden is not None:
            return self.model(X, (hidden, cell))
        else:
            return self.model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)[0]  # drops the hidden state because it's not needed during batch operations, only during single-frame inference
        loss = self.criterion(preds, y)
        # perform logging
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)[0]  # drops the hidden state because it's not needed during batch operations, only during single-frame inference
        loss = self.criterion(preds, y)
        # perform logging
        self.log("val_loss", loss, prog_bar=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--landmarks', type=int, default=68, help='Number of landmarks to predict during training')
    parser.add_argument('--optimizer', default='adam', help='Type of optimizer to use. Adam or SGD')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Samples per batch')
    parser.add_argument('--layers', type=int, default=4, help='Number of LSTM layers')
    parser.add_argument('--hidden-size', type=int, default=256, help='Number of units per LSTM layer. Must be >= 2 * landmarks')
    parser.add_argument('--save-path', required=True, help='Where to save the final model')
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs for training')
    parser.add_argument('--log-every-n-steps', type=int, default=250, help='Log to Weights & Biases after N steps')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.optimizer = args.optimizer.lower()
    assert args.landmarks == 68  # TODO: Someday, maybe support other number of landmarks
    assert args.optimizer in {'adam', 'sgd'}
    assert args.hidden_size >= 2 * args.landmarks
    """
    args = SimpleNamespace(
        num_landmarks=136,
        optimizer='adam',
        learning_rate=1e-3,
        batch_size=128,
        save_path='./pytorch_output/',
        # Trainer args
        gpus=int(torch.cuda.is_available()),
        epochs=50,
    )
    """
    # os.environ["WANDB_START_METHOD"] = "thread"
    os.environ['WANDB_DISABLE_CODE']='True'
    # # Instantiate Model
    model = TalkingFaceLSTM(num_landmarks = args.landmarks,
                            optimizer = args.optimizer, 
                            lr = args.learning_rate,
                            layers=args.layers,
                            hidden_size=args.hidden_size,
                            )
    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': int(torch.cuda.is_available()), 'max_epochs': args.epochs}
    wandb_logger = WandbLogger(project="audio-vtuber", job_type="train", log_model=True)
    wandb_logger.watch(model, log_freq=max(100, args.log_every_n_steps))
    wandb_logger.log_hyperparams(vars(args))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **trainer_args)
    trainer.fit(model, datamodule=GridDataModule(batch_size=args.batch_size))
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)