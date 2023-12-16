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
class CoNLL2000Chunking_Dataset(pl.LightningDataModule):

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
        # fields = [('text', self.TEXT), ('label', self.LABEL)]
        fields = []
        t1 = ('text', self.TEXT)
        t2 = ('label', self.LABEL)
        fields.append(('text_a', self.TEXT))
        fields.append(('text_b', self.TEXT))
        fields.append(('label', self.LABEL))

        self.train_dataset, self.valid_dataset, self.test_dataset = datasets.CoNLL2000Chunking.splits(fields=fields, root='../.data/chunk')


        # build the vocabulary(self.TEXT, self.LABEL)
        self.TEXT.build_vocab(self.train_dataset)
        self.LABEL.build_vocab(self.train_dataset)

        self.vocab = self.TEXT.vocab
        self.vocab_size = len(self.vocab)
        self.label_size = len(self.LABEL.vocab)
        self.pad_index = self.vocab.stoi['<pad>']
        self.unk_index = self.vocab.stoi['<unk>']
        self.train_dataset.example

    def setup(self):
        pass

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


if __name__ == '__main__':
    a = CoNLL2000Chunking_Dataset(128)
    a.prepare_data()
    a.setup()
    print(a.vocab_size)
    print(a.label_size)
    print(len(a.train_dataset))
    print(len(a.valid_dataset))
    print(len(a.test_dataset))
    b = a.test_dataset
    c = a.test_dataloader()
    for each in a.train_dataloader():
        print(each)
        break
    # print(a.train_d