from torchtext import data, datasets
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import os
import pickle


class AgNews_Dataset(pl.LightningDataModule):

    def __init__(self, hparams):
        if isinstance(hparams, int):  # 判断hparams(即args)是否是int类型
            self.batch_size = hparams  # 定义batch_size
        else:
            self.hparams = hparams  # 将传入的hparams(即args)初始化自身的hparams
            self.batch_size = self.hparams.batch_size  # 定义batch_size
        self.base_dir = os.path.dirname(__file__)  # 返回py文件的绝对路径('D:\\zip\\embedding\\datasets')
        self.cache_dir = os.path.join(self.base_dir, 'cache')  # 拼接地址('D:\\zip\\embedding\\datasets\\cache')

    def prepare_data(self):
        PICKLE_FILE = os.path.join(self.cache_dir,
                                   'ag_news.pkl')  # 拼接地址('D:\\zip\\embedding\\datasets\\cache\\ag_news.pkl')
        if os.path.exists(PICKLE_FILE):  # 判断当前路径是否存在
            # 加载获取训练数据集和测试数据集
            with open(PICKLE_FILE, 'rb') as f:  # 以二进制格式打开文件用于只读
                _ = pickle.load(f)  # 将二进制对象文件转化为python对象
                self.train_dataset = _['train']  # 获得训练集
                self.test_dataset = _['test']  # 获得数据集
        else:
            # 没有当前数据则先下载再获取
            self.train_dataset, self.test_dataset = datasets.AG_NEWS(ngrams=1, include_unk=True)  # 下载获取数据集
            with open(PICKLE_FILE, 'wb') as f:  # 以二进制格式打开文件只用于写入，已存在则覆盖，不存在则创建新文件
                pickle.dump({  # 将训练集和数据集保存到文件f(即D:\\zip\\embedding\\datasets\\cache\\ag_news.pkl)中
                    'train': self.train_dataset,
                    'test': self.test_dataset
                }, f)

        self.vocab = self.train_dataset.get_vocab()  # 统计train_dataset中不同单词总数
        self.label_size = len(self.train_dataset.get_labels())  # 获得文本分类总数
        self.vocab_size = len(self.vocab)  # 获得单词总数个数
        self.pad_index = self.vocab.stoi['<pad>']  # batch里句子填充的符号
        self.unk_index = self.vocab.stoi['<unk>']  # batch里单词未知的符号

        split_length = int(len(self.train_dataset) * 0.8)  # 以8:2把训练集拆分为训练集和验证集
        self.train_dataset, self.valid_dataset = random_split(self.train_dataset,
                                                              [split_length,
                                                               len(self.train_dataset) - split_length])  # 随机分配训练集和验证集

    def setup(self):
        pass

    # 暂时:自定义如何取样本，用于下方dataloader
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
    a = AgNews_Dataset(128)
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
