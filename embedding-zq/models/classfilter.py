from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attn import *
from datasets import *
from pytorch_lightning.loggers import TensorBoardLogger


class Classifier(pl.LightningModule):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.policy = policy()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(input_dim, args.embedding_dim)
        self.rnn = nn.RNN(args.embedding_dim, args.hidden_dim, bidirectional=args.bi, batch_first=True)
        self.fc = torch.nn.Sequential(
            nn.Linear(args.hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
        nn.Linear(args.hidden_dim, output_dim)
        self.learning_rate = args.learning_rate
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, text):
        x, x_len = text
        embedded1 = self.embedding(x)
        embedded2 = self.embedding(x)

        packed = pack_padded_sequence(embedded1, x_len, batch_first=True)
        output, hidden = self.rnn(packed)
        unpacked_output, length = pad_packed_sequence(output, batch_first=True)
        _ = self.fc(unpacked_output)
        return _

    def training_step(self, batch, batch_idx):
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)
        acc = self.valid_acc(torch.max(y_hat, 1)[1], y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        self.valid_acc(torch.max(y_hat, 1)[1], y)

    def test_step(self, batch, batch_idx):
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.test_acc(torch.max(y_hat, 1)[1], y)

    def validation_epoch_end(self, outs):
        self.log('valid_acc', self.valid_acc.compute())

    def test_epoch_end(self, outs):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=self.hparams['args'].learning_rate)
        return torch.optim.Adam(params, lr=self.hparams['args'].learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--learning_rate', type=float, default=0.003)
        return parser

# 非负的最小值，使得归一化时分母不为0
