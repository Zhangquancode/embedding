from torchtext import datasets
from torchtext.legacy import data
import pytorch_lightning as pl
import os


class SQuAD_Dataset(pl.LightningDataModule):

    def __init__(self, hparams):
        if isinstance(hparams, int):
            self.batch_size = hparams
        else:
            self.hparams = hparams
            self.batch_size = self.hparams.batch_size
        self.base_dir = os.path.dirname(__file__)

    def has_prepare_data(self):

        # set up fields
        # self.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        # self.LABEL = data.Field(sequential=False)

        self.context, self.question, self.answers, self.answer_start = datasets.SQuAD1(split=('train', 'dev'))

        # build the vocabulary
        self.TEXT.build_vocab(self.train_dataset)
        self.LABEL.build_vocab(self.train_dataset)

        # split_length = int(len(self.train_dataset) * 0.8)
        self.train_dataset, self.valid_dataset = self.train_dataset.split(split_ratio=0.8)


        self.vocab = self.TEXT.vocab
        self.vocab_size = len(self.vocab)
        self.label_size = len(self.LABEL.vocab)
        self.pad_index = self.vocab.stoi['<pad>']
        self.unk_index = self.vocab.stoi['<unk>']

    def setup(self):
        pass

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
    a = SQuAD_Dataset(32)
    a.has_prepare_data()
    a.setup()
    for each in a.train_dataloader():
        print(each.text)
        break
    # for each in a.train_dataloader():
    #     print(each)
    #     break
