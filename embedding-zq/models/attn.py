import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, rnn_type, drop):
        super(Attention, self).__init__()
        if rnn_type == 'RNN':
            self.rnncell = nn.RNNCell(embedding_dim, hidden_dim)
            # self.rnn = nn.RNN(embedding_dim, hidden_dim)
        elif rnn_type == 'GRU':
            self.rnncell = nn.GRUCell(embedding_dim, hidden_dim)
        elif rnn_type == 'LSTM':
            self.rnncell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.W_query = nn.Linear(embedding_dim, embedding_dim)
        # self.W_key1 = nn.Linear(embedding_dim, embedding_dim)
        # self.W_key2 = nn.Linear(embedding_dim, embedding_dim)
        # self.W_value = nn.Linear(embedding_dim, embedding_dim)
        # self.scale_factor = torch.sqrt(torch.tensor(embedding_dim).float())
        self.dropout = nn.Dropout(p=0.1)
        # self.n_distance = 0

    def forward(self, e1, e2, e3, batch_size, x_len, policy, layer, model, drop, rnn_type, ):

        # if layer != 1:
        #     e1, e2, e3 = self.embedding_attn(e1, e2, e3)

        # 进行矩阵转置(64,74,200)-->(74,64,200)
        e1 = torch.transpose(e1, dim0=1, dim1=0)
        e2 = torch.transpose(e2, dim0=1, dim1=0)
        e3 = torch.transpose(e3, dim0=1, dim1=0)
        # n = len(e1)
        # self.n_distance = 0

        # length_list = []
        # for t1 in e1:
        #     length_list.append(len(t1))
        # length_list = torch.tensor(length_list)
        # todo embedding_layer = 1
        if layer == 1:
            index = len(e1[0][0])
            h = torch.zeros(batch_size, index).cuda()
            i = 0
            for test1, test2, test3 in zip(e1, e2, e3):  # test1 nn.embedding, test2 glove, test3 gensim
                if model == 1:
                    test = test1
                elif model == 2:
                    test = test2
                elif model == 3:
                    test = test3
                if i == 0:
                    # test = self.dropout(test, 0.5)
                    h = self.rnncell(test)
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        # h = self.relu(h)
                        output = h.unsqueeze(1)

                    elif rnn_type == 'LSTM':
                        output = h[0].unsqueeze(1)
                else:
                    # test = self.dropout(test, 0.5)
                    h = self.rnncell(test, h)
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        # h = self.relu(h)
                        output = torch.cat((output, h.unsqueeze(1)), dim=1)

                    elif rnn_type == 'LSTM':
                        output = torch.cat((output, h[0].unsqueeze(1)), dim=1)
                i += 1

        # todo embedding_layer = 2
        elif layer == 2:
            # 设置初始权重
            # {Tensor:(1,2)}tensor([[0.5,0.5]])
            e_index = torch.tensor([[1 / 2, 1 / 2]]).cuda()
            # 获得词向量维度
            # index = len(e1[0][0])
            # {Tensor:(200,200)}
            # {Tensor:(64,200)}
            # ???:这里的h或许是论文中的h
            # h = torch.zeros(batch_size, index).cuda()
            i = 0
            for test1, test2, test3 in zip(e1, e2, e3):

                merge = torch.cat((test1, test3), dim=1)
                options = torch.stack((test1, test3), dim=1)
                if i == 0:
                    h = torch.einsum('ik, akj -> aj', e_index, options)
                    # 将初始状态重叠
                    merge = torch.cat((h, merge), dim=1)
                    # 将环境送入policy进行选择
                    action = policy.choose(merge.detach())
                    # 存储当前的环境和选择
                    policy.store_transtion(merge.detach(), action)
                    # 得到选择后的词向量
                    merge = torch.einsum('ik, ikj -> ij', action, options)
                    # rnn
                    h = self.rnncell(merge)

                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        # 激活函数
                        h = self.relu(h)
                        #
                        output = h.unsqueeze(1)

                    elif rnn_type == 'LSTM':
                        output = h[0].unsqueeze(1)

                else:
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        merge = torch.cat((h, merge), dim=1)
                        # 将环境送入policy进行选择
                        action = policy.choose(merge.detach())
                        # 存储当前的环境和选择
                        policy.store_transtion(merge.detach(), action)
                        # 得到选择后的词向量
                        merge = torch.einsum('ik, ikj -> ij', action, options)
                        # rnn
                        h = self.rnncell(merge, h)
                        # 激活函数
                        h = self.relu(h)
                        # 将通过rnn的词向量合在一起
                        output = torch.cat((output, h.unsqueeze(1)), dim=1)

                    elif rnn_type == 'LSTM':
                        merge = torch.cat((h[0], merge), dim=1)
                        # 将环境送入policy进行选择
                        action = policy.choose(merge.detach())
                        # 存储当前的环境和选择
                        policy.store_transtion(merge.detach(), action)
                        # 得到选择后的词向量
                        merge = torch.einsum('ik, ikj -> ij', action, options)
                        h = self.rnncell(merge, h)
                        output = torch.cat((output, h[0].unsqueeze(1)), dim=1)

                i += 1

        # todo embedding_layer = 3
        elif layer == 3:
            # 设置初始权重
            # {Tensor:(1,3)}tensor([[0.3333,0.3333,0.3333]])
            e_index = torch.tensor([[1 / 3, 1 / 3, 1 / 3]]).cuda()

            # 获得词向量维度
            # index = len(e1[0][0])
            # {Tensor:(200,200)}
            # {Tensor:(64,200)}
            # ???:这里的h或许是论文中的h
            # h = torch.zeros(batch_size, index)#.cuda()
            i = 0
            for test1, test2, test3 in zip(e1, e2, e3):

                merge = torch.cat((test1, test2), dim=1)
                merge = torch.cat((merge, test3), dim=1)
                options = torch.stack((test1, test2), dim=1)
                options = torch.cat((options, test3.unsqueeze(1)), dim=1)

                if i == 0:
                    h = torch.einsum('ik, akj -> aj', e_index, options)
                    merge = torch.cat((h, merge), dim=1)
                    # 将环境送入policy进行选择
                    action = policy.choose(merge.detach())
                    # action, action_index = policy.choose(merge)
                    # 存储当前的环境和选择
                    policy.store_transtion(merge.detach(), action)
                    # 得到选择后的词向量
                    merge = torch.einsum('ik, ikj -> ij', action, options)
                    h = self.rnncell(merge)
                    # options = torch.einsum('ik, akj -> aj', action, options)
                    # h = self.rnncell(options)
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        h = self.relu(h)
                        output = h.unsqueeze(1)

                    elif rnn_type == 'LSTM':
                        output = h[0].unsqueeze(1)
                else:
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        merge = torch.cat((h, merge), dim=1)
                        # 将环境送入policy进行选择
                        action = policy.choose(merge.detach())
                        # action, action_index = policy.choose(h)
                        # merge = h
                        # 存储当前的环境和选择
                        policy.store_transtion(merge.detach(), action)
                        # 得到选择后的词向量
                        merge = torch.einsum('ik, ikj -> ij', action, options)
                        h = self.rnncell(merge, h)
                        # options = torch.einsum('ik, akj -> aj', action, options)
                        # h = self.rnncell(options, h)
                        h = self.relu(h)
                        # 存储当前的环境和选择
                        # if (policy.EPSILON != 1):
                        #     policy.store_transtion(merge, action_index, h)
                        output = torch.cat((output, h.unsqueeze(1)), dim=1)

                    elif rnn_type == 'LSTM':
                        merge = torch.cat((h[0], merge), dim=1)
                        # 将环境送入policy进行选择
                        action = policy.choose(merge.detach())
                        # action, action_index = policy.choose(h)
                        # merge = h
                        # 存储当前的环境和选择
                        policy.store_transtion(merge.detach(), action)
                        # 得到选择后的词向量
                        merge = torch.einsum('ik, ikj -> ij', action, options)
                        h = self.rnncell(merge, h)
                        output = torch.cat((output, h[0].unsqueeze(1)), dim=1)
                i += 1
        # self.n_distance /= n
        # x_len = torch.tensor(x_len).cuda()
        # output = self.dropout(output)
        result = [output[i, :x_len[i], :].sum(dim=0) / x_len[i] for i in range(0, batch_size)]
        # result = [sub_tensor.sum(dim=0) for sub_tensor in result]
        # result = [sub_tensor / x_len[i].float() for i, sub_tensor in enumerate(result)]
        wait_output = torch.stack(result)
        # wait_output = torch.einsum('ble, b->be', output, 1 / x_len)
        return wait_output
