from torchtext import data, datasets
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import os
import pickle


class merge(pl.LightningDataModule):

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

        self.train_dataset, self.test_dataset = data.TabularDataset.splits(
            path='.data/merge',
            format='csv',
            train='train.csv', test='test.csv',
            fields=[
                ('label', self.LABEL),
                ('text', self.TEXT)
            ]
        )
        # self.train_dataset, self.test_dataset = datasets.IMDB.splits(self.TEXT, self.LABEL)

        # build the vocabulary
        self.TEXT.build_vocab(self.train_dataset, min_freq=5)
        self.LABEL.build_vocab(self.train_dataset)

        self.train_dataset, self.valid_dataset = self.train_dataset.split(split_ratio=0.8)

        self.vocab = self.TEXT.vocab
        self.vocab_size = len(self.vocab)
        self.label_size = len(self.LABEL.vocab)
        self.pad_index = self.vocab.stoi['<pad>']
        self.unk_index = self.vocab.stoi['<unk>']

    def setup(self):
        pass

    # 暂时:自定义如何取样本，用于下方dataloader
    # def collate_fn(self, batch):
    #     batch.sort(key=lambda x: len(x.text), reverse=True)
    #     labels = [int(_.label) for _ in batch]
    #     labels = torch.LongTensor(labels)
    #     seq_length = [len(_.text) for _ in batch]
    #     data_seq = [_.text for _ in batch]
    #     data_seq = pad_sequence(data_seq, batch_first=True, padding_value=self.pad_index)
    #     return data_seq, seq_length, labels

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
    a = merge(128)
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
    # print(a.train_dataset.get_labels())

    # for each in a.train_dataloader():
    #     print(each)
    #     break
