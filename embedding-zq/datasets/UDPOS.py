import pandas as pd
import numpy as np
import typing
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk import word_tokenize, sent_tokenize
from torchtext import data, datasets
from typing import Union
class UDPOS_Dataset(pl.LightningDataModule):

    def __init__(self, hparams):
        if isinstance(hparams, int):
            self.batch_size = hparams
        else:
            self.hparams = hparams
            self.batch_size = self.hparams.batch_size


    def prepare_data(self):

        labeltokenize = lambda x: [int(x)]
        self.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        self.LABEL = data.Field(sequential=False, tokenize=labeltokenize, unk_token=None)

        # self.train_dataset, self.test_dataset = data.TabularDataset.splits(
        #     path='.data/chunk',
        #     format= 'txt',
        #     train='train.txt', test='test.txt',
        #     fields=[
        #         ('label', self.LABEL),
        #         ('text', self.TEXT)
        #     ]
        # )

        self.train_dataset, self.valid_dataset, self.test_dataset = datasets.UDPOS.splits((self.TEXT, self.LABEL), root='../.data')


        # build the vocabulary(self.TEXT, self.LABEL)
        self.TEXT.build_vocab(self.train_dataset)
        self.LABEL.build_vocab(self.train_dataset)

        self.vocab = self.TEXT.vocab
        self.vocab_size = len(self.vocab)
        self.label_size = len(self.LABEL.vocab)
        self.pad_index = self.vocab.stoi['<pad>']
        self.unk_index = self.vocab.stoi['<unk>']

    def setup(self):
        pass

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        labels = [_[0] for _ in batch]
        labels = torch.LongTensor(labels)
        seq_length = [len(_[1]) for _ in batch]
        data_seq = [_[1] for _ in batch]
        data_seq = pad_sequence(data_seq, batch_first=True, padding_value=self.pad_index)
        return data_seq, seq_length, labels

    def train_dataloader(self):
        return data.BucketIterator(self.train_dataset, self.batch_size, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True)

    def val_dataloader(self):
        return data.BucketIterator(self.valid_dataset, self.batch_size, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True)

    def test_dataloader(self):
        return data.BucketIterator(self.test_dataset, self.batch_size, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True)
    def transfer_batch_to_device(self, batch, device):
        x, x_len = batch.text
        x = x.to(device)
        y = batch.label.to(device)
        return x, x_len, y


if __name__ == '__main__':
    a = UDPOS_Dataset(32)
    a.prepare_data()
    a.setup()
    for each in a.train_dataloader():
        # print(each.text)
        break