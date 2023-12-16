from argparse import ArgumentParser
import pytorch_lightning as pl
from models.attn import *
from models.policy import *
import torch.nn as nn
# from models.policy2 import *
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LitClassifier(pl.LightningModule):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.save_hyperparameters()
        # 数据集中出现的全部单词
        self.vocab_list = args.vocab_list
        # pytorch自带embedding模型
        self.embedding1 = nn.Embedding(input_dim, args.embedding_dim)
        self.embedding1.weight.data.uniform_(-1, 1)
        self.embedding1.weight.requires_grad = True

        # glove
        # self.glove_embeddings_dict = {}
        # with open("./glove/glove.6B." + str(args.embedding_dim) + "d.txt", 'r', encoding="utf-8") as f:
        #     for line in f:
        #         values = line.split()
        #         word = values[0]
        #         vector = np.asarray(values[1:], "float32")
        #         self.glove_embeddings_dict[word] = vector
        self.embedding2 = nn.Embedding(input_dim, args.embedding_dim)
        self.embedding2.weight.data.normal_(0, 0.1)

        # glove_vector = []
        # for word in self.vocab_list:
        #     if word in self.glove_embeddings_dict:
        #         glove_vector.append(self.glove_embeddings_dict[word])
        #     else:
        #         glove_vector.append(np.random.randn(args.embedding_dim))
        #
        # glove_vector = np.array(glove_vector)
        # self.embedding2.weight.data.copy_(torch.from_numpy(glove_vector))
        self.embedding2.weight.requires_grad = True  # 是否随着网络进行更新

        # gensim
        # self.gensim_embeddings_dict = {}
        # with open("./models/w2cmodel/word2vec_" + str(args.embedding_dim) + ".txt", 'r', encoding="utf-8") as f:
        #     for line in f:
        #         values2 = line.split()
        #         word2 = values2[0]
        #         vector2 = np.asarray(values2[1:], "float32")
        #         self.gensim_embeddings_dict[word2] = vector2
        self.embedding3 = nn.Embedding(input_dim, args.embedding_dim)
        vector = np.random.randn(input_dim, args.embedding_dim)
        self.embedding3.weight.data.copy_(torch.from_numpy(vector))

        # gensim_vector = []
        # for word in self.vocab_list:
        #     if word in self.gensim_embeddings_dict:
        #         gensim_vector.append(self.gensim_embeddings_dict[word])
        #     else:
        #         gensim_vector.append(np.random.randn(args.embedding_dim))
        # gensim_vector = np.array(gensim_vector)
        #
        # self.embedding3.weight.data.copy_(torch.from_numpy(gensim_vector))
        self.embedding3.weight.requires_grad = True  # 是否随着网络进行更新
        self.policy = Policy(args.embedding_dim, args.embedding_layer, args.Gamma, args.dropout)
        # self.policy = DQN(args.embedding_dim, args.embedding_layer, args.batch_size, args.LR, args.epsilon, args.gamma, args.target_replace_iter, args.memory_capacity)
        # self.rnncell = nn.RNNCell(args.embedding_dim, args.hidden_dim)
        # self.rnn = nn.RNN(args.embedding_dim, args.hidden_dim)
        # 注意力层，筛选embedding
        # self.relu = nn.ReLU()
        self.attn = Attention(args.embedding_dim, args.hidden_dim, args.rnn_type, args.dropout)

        # 全链接层，输出结果
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

        self.learning_rate = args.learning_rate

        self.rnn_type = args.rnn_type
        self.dropout = args.dropout
        self.embedding_layer = args.embedding_layer
        self.embedding_model = args.embedding_model
        self.atten_TF = args.atten
        self.Dropout = nn.Dropout(args.dropout, inplace=True)

        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.total_reward = 0
        # self.lambda1 = 1
        # self.lambda2 = 0

    def forward(self, text):
        x, x_len = text
        e1 = self.embedding1(x)
        e2 = self.embedding2(x)
        e3 = self.embedding3(x)

        wait_output = self.attn(e1, e2, e3,  # e1 nn.embedding, e2 glove, e3 gensim
                                len(x),  # batch_size
                                x_len,  # 处理RNNcell返回值
                                self.policy,  # policy决策函数
                                self.embedding_layer,  # embedding模型个数
                                self.embedding_model,  # 当使用1个embedding时，选择第几个模型
                                self.dropout,  # 多embedding模型时embedding得分dropout概率
                                self.rnn_type  # cell模型类别
                                )
        _ = self.fc(wait_output)
        return _

    def training_step(self, batch, batch_idx):
        self.policy.obs, self.policy.acts, self.policy.rewards = [], [], []
        # y, (x, x_len) = batch.label, batch.text #yahoo数据集

        self.policy.exp = True
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss2 = 0
        if (self.embedding_layer != 1):
            self.policy.store_reward(y_hat.detach(), y)
            loss2, self.total_reward = self.policy.learn(x_len)
        loss1 = F.cross_entropy(y_hat, y)
        loss = loss1 + loss2
        acc = self.valid_acc(torch.max(y_hat, 1)[1], y)
        self.log('reward', self.total_reward, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss1', loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss2', loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.policy.obs, self.policy.acts, self.policy.rewards = [], [], []
        # y, (x, x_len) = batch.label, batch.text  # yahoo数据集

        self.policy.exp = False
        x, x_len, y = batch
        y_hat = self((x, x_len))
        # if (self.embedding_layer != 1):
        #     self.policy.store_reward(torch.max(y_hat, 1)[1], y)
        loss = F.cross_entropy(y_hat, y)
        # 清除当前游戏字典
        self.policy.obs, self.policy.acts, self.policy.rewards = [], [], []
        # self.policy.probs = []
        self.log('valid_loss', loss)
        self.valid_acc(torch.max(y_hat, 1)[1], y)


    def test_step(self, batch, batch_idx):
        self.policy.obs, self.policy.acts, self.policy.rewards = [], [], []
        # y, (x, x_len) = batch.label, batch.text  # yahoo数据集

        # self.policy.EPSILON = 1
        self.policy.exp = False
        self.policy.pt = True
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)

        # idx = torch.where(x == 1445)
        # choose = [self.policy.acts[idx[1][i]][idx[0][i]] for i in range(len(idx[0]))]

        self.policy.obs, self.policy.acts, self.policy.rewards = [], [], []
        # self.policy.probs = []
        self.log('test_loss', loss)
        self.test_acc(torch.max(y_hat, 1)[1], y)

        return (y.eq(torch.max(y_hat, 1)[1]) + 0).sum() / len(y)

    def validation_epoch_end(self, outs):
        self.log('valid_acc', self.valid_acc.compute())

    def test_epoch_end(self, outs):
        acc = sum(outs) / len(outs)
        print('=============== RESULT ===================')
        print(acc)
        print('=============== RESULT ===================')

        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        # params = filter(lambda p: p.requires_grad, self.parameters())
        # # return torch.optim.Adam(self.parameters(), lr=self.hparams['args'].learning_rate)
        # return torch.optim.Adam(params, lr=self.hparams['args'].learning_rate)
        params1 = []
        params2 = []
        named_parameters = self.named_parameters()
        for name, param in named_parameters:
            if name.startswith('policy'):
                params2.append(param)
            else:
                params1.append(param)
        optimizer = torch.optim.Adam(params1, lr=self.hparams['args'].learning_rate)
        optimizer.add_param_group({'params': params2, 'lr': self.hparams['args'].rl_rate})

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--learning_rate', type=float, default=0.0005)
        parser.add_argument('--rl_rate', type=float, default=0.00001)
        # parser.add_argument('--learning_rate_policy', type=float, default=0.0008)
        parser.add_argument('--Gamma', type=float, default=0.99)
        return parser
