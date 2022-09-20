from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.optim import Adam, SGD
import torch.nn as nn

from dataset import GridDataModule


class TalkingFaceLSTM(pl.LightningModule):
    def __init__(self, num_landmarks,
                optimizer='adam', lr=1e-3):
        super().__init__()

        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        self.num_landmarks = num_landmarks
        self.lr = lr
        #instantiate loss criterion
        self.criterion = nn.MSELoss()
        self.model = nn.LSTM(input_size=128, hidden_size=256, proj_size=136, num_layers=4, dropout=0.2, batch_first=True, )
        # self.train_accuracy = torchmetrics.Accuracy()

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)[0]
        loss = self.criterion(preds, y)
        # perform logging
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)[0]
        loss = self.criterion(preds, y)
        # perform logging
        self.log("val_loss", loss, prog_bar=True)


if __name__ == '__main__':
    args = SimpleNamespace(
        num_landmarks=136,
        optimizer='adam',
        learning_rate=1e-3,
        batch_size=128,
        save_path='./pytorch_output',
        # Trainer args
        gpus=int(torch.cuda.is_available()),
        num_epochs=200,
    )

    # # Instantiate Model
    model = TalkingFaceLSTM(num_landmarks = args.num_landmarks,
                            optimizer = args.optimizer, 
                            lr = args.learning_rate,
                            )
    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs}
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, datamodule=GridDataModule(batch_size=128))
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)