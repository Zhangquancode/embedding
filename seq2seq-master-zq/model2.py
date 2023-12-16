import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# from utils import reduce_dimension
# class Encoder(nn.Module):
#     def __init__(self, input_size, embed_size, hidden_size,
#                  n_layers=1, dropout=0.5):
#         super(Encoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.embed_size = embed_size
#         self.embed = nn.Embedding(input_size, embed_size)
#         self.gru = nn.GRU(embed_size, hidden_size, n_layers,
#                           dropout=dropout, bidirectional=False)
#
#     def forward(self, src, hidden=None):
#         embedded = self.embed(src)
#         outputs, hidden = self.gru(embedded, hidden)
#         # sum bidirectional outputs
#         # outputs = (outputs[:, :, :self.hidden_size] +
#         #            outputs[:, :, self.hidden_size:])
#         return outputs, hidden
SOS_token = 0 # 开始标志
EOS_token = 1 # 结束标志
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding1 = nn.Embedding(input_size, embed_size)

        # glove
        embeddings_dict = {}
        with open("../embedding/glove/glove.6B." + str(embed_size) + "d.txt", 'r', encoding="utf-8") as f:
        # with open("./glove/glove.6B."+str(embed_size)+"d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        self.embedding2 = nn.Embedding(input_size, embed_size)
        self.embedding2.weight.data.copy_(torch.from_numpy(vector))
        self.embedding2.weight.requires_grad = True #是否随着网络进行更新


        # gensim
        embeddings_dict2 = {}
        with open("../embedding/models/w2cmodel/word2vec_" + str(embed_size) + ".txt", 'r', encoding="utf-8") as f:
        # with open("./w2cmodel/word2vec_"+str(embed_size)+".txt", 'r', encoding="utf-8") as f:
            for line in f:
                values2 = line.split()
                word2 = values2[0]
                vector2 = np.asarray(values2[1:], "float32")
                embeddings_dict2[word2] = vector2
        self.embedding3 = nn.Embedding(input_size, embed_size)
        self.embedding3.weight.data.copy_(torch.from_numpy(vector2))
        self.embedding3.weight.requires_grad = True #是否随着网络进行更新
        # self.rnncell = nn.GRUCell(embed_size, hidden_size)
        self.rnncell = nn.RNNCell(embed_size, hidden_size)
        # self.rnncell = nn.LSTMCell(embed_size, hidden_size)


    def forward(self, src, hidden=None):
        model = 1
        batch_size = 64
        rnn_type = 'RNN'#LSTM, GRU, RNN
        layer = 1
        drop = 0.2
        e1 = self.embedding1(src)
        e2 = self.embedding2(src)
        e3 = self.embedding3(src)


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
        elif layer == 2:
            i = 0
            e_index = torch.tensor([[1 / 2, 1 / 2]]).cuda()
            index = len(e1[0][0])
            e1_index = torch.zeros(index, index).cuda()
            e2_index = torch.zeros(index, index).cuda()
            for i in range(index):
                if i % 2 == 0:
                    e1_index[i][i] = 1
                else:
                    e2_index[i][i] = 1
            h = torch.zeros(batch_size, index).cuda()
            i = 0
            for test1, test2, test3 in zip(e1, e2, e3):
                test1 = torch.einsum('ik, kj -> ij', test1, e1_index)
                test2 = torch.einsum('ik, kj -> ij', test2, e2_index)
                test = torch.stack((test1, test2), dim=1)

                if i == 0:
                    test = torch.einsum('ik, akj -> aj', e_index, test)
                    h = self.rnncell(test)

                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        # h = self.relu(h)
                        output = h.unsqueeze(1)

                    elif rnn_type == 'LSTM':
                        output = h[0].unsqueeze(1)

                else:
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        scores = torch.einsum('ij, ikj -> ik', h, test)
                        p_attn2 = self.dropout(scores, drop)
                        p_attn = F.softmax(p_attn2, dim=-1)
                        test = torch.einsum('ij, ijk -> ik', p_attn, test)
                        h = self.rnncell(test, h)
                        # h = self.relu(h)
                        output = torch.cat((output, h.unsqueeze(1)), dim=1)

                    elif rnn_type == 'LSTM':
                        scores = torch.einsum('ij, ikj -> ik', h[0], test)
                        p_attn2 = self.dropout(scores, drop)
                        p_attn = F.softmax(p_attn2, dim=-1)
                        test = torch.einsum('ij, ijk -> ik', p_attn, test)
                        h = self.rnncell(test, h)
                        output = torch.cat((output, h[0].unsqueeze(1)), dim=1)

                i += 1
            # todo embedding_layer = 3
        elif layer == 3:
            i = 0
            e_index = torch.tensor([[1 / 3, 1 / 3, 1 / 3]]).cuda()
            index = len(e1[0][0])
            e1_index = torch.zeros(index, index).cuda()
            e2_index = torch.zeros(index, index).cuda()
            e3_index = torch.zeros(index, index).cuda()
            for i in range(index):
                if i % 3 == 0:
                    e1_index[i][i] = 1
                if (i + 2) % 3 == 0:
                    e2_index[i][i] = 1
                if (i + 1) % 3 == 0:
                    e3_index[i][i] = 1
            h = torch.zeros(batch_size, index).cuda()
            i = 0
            for test1, test2, test3 in zip(e1, e2, e3):

                test1 = torch.einsum('ik, kj -> ij', test1, e1_index)
                test2 = torch.einsum('ik, kj -> ij', test2, e2_index)
                test3 = torch.einsum('ik, kj -> ij', test3, e3_index)
                test = torch.stack((test1, test2), dim=1)
                test = torch.cat((test, test3.unsqueeze(1)), dim=1)

                if i == 0:
                    test = torch.einsum('ik, akj -> aj', e_index, test)
                    h = self.rnncell(test)
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        # h = self.relu(h)
                        output = h.unsqueeze(1)

                    elif rnn_type == 'LSTM':
                        output = h[0].unsqueeze(1)
                else:
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        scores = torch.einsum('ij, ikj -> ik', h, test)
                        p_attn2 = self.dropout(scores, drop)
                        p_attn = F.softmax(p_attn2, dim=-1)
                        test = torch.einsum('ij, ijk -> ik', p_attn, test)
                        h = self.rnncell(test, h)
                        # h = self.relu(h)
                        output = torch.cat((output, h.unsqueeze(1)), dim=1)

                    elif rnn_type == 'LSTM':
                        scores = torch.einsum('ij, ikj -> ik', h[0], test)
                        p_attn2 = self.dropout(scores, drop)
                        p_attn = F.softmax(p_attn2, dim=-1)
                        test = torch.einsum('ij, ijk -> ik', p_attn, test)
                        h = self.rnncell(test, h)
                        output = torch.cat((output, h[0].unsqueeze(1)), dim=1)
                i += 1
        output = torch.transpose(output, dim0=1, dim1=0)

        if rnn_type == 'RNN' or rnn_type == 'GRU':
            return output, h.unsqueeze(0)
        elif rnn_type == 'LSTM':
            return output, h[0].unsqueeze(0)

    def dropout(self, X, drop_prob):
        X = X.float().cuda()
        assert 0 <= drop_prob <= 1
        keep_prob = 1 - drop_prob
        if keep_prob == 0:
            return torch.zeros_like(X)
        mask = (torch.randn(X.shape) < keep_prob).float().cuda()
        return mask * X / keep_prob


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embbed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()

        # 设置参数
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = n_layers

        # 以适当的维度，初始化每一层。
        # decoder层由embedding, GRU, 线性层和Log softmax 激活函数组成
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)  # 线性层
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax函数

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


# class Decoder(nn.Module):
#     def __init__(self, embed_size, hidden_size, output_size,
#                  n_layers=1, dropout=0.2):
#         super(Decoder, self).__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#
#         self.embed = nn.Embedding(output_size, embed_size)
#         self.dropout = nn.Dropout(dropout, inplace=True)
#         self.attention = Attention(hidden_size)
#         self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
#                           n_layers, dropout=dropout)
#         self.out = nn.Linear(hidden_size * 2, output_size)
#
#     def forward(self, input, last_hidden, encoder_outputs):
#         # Get the embedding of the current input word (last output word)
#         embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
#         embedded = self.dropout(embedded)
#         # Calculate attention weights and apply to encoder outputs
#         attn_weights = self.attention(last_hidden[-1], encoder_outputs)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
#         context = context.transpose(0, 1)  # (1,B,N)
#         # Combine embedded input word and attended context, run through RNN
#         rnn_input = torch.cat([embedded, context], 2)
#         output, hidden = self.gru(rnn_input, last_hidden)
#         output = output.squeeze(0)  # (1,B,N) -> (B,N)
#         context = context.squeeze(0)
#         output = self.out(torch.cat([output, context], 1))
#         output = F.log_softmax(output, dim=1)
#         return output, hidden, attn_weights

# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size = src.size(1)
#         max_len = trg.size(0)
#         vocab_size = self.decoder.output_size
#         outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
#
#
#         encoder_output, hidden = self.encoder(src)
#         hidden = hidden[:self.decoder.n_layers]
#         output = Variable(trg.data[0, :])  # sos
#         for t in range(1, max_len):
#             output, hidden, attn_weights = self.decoder(
#                     output, hidden, encoder_output)
#             outputs[t] = output
#             is_teacher = random.random() < teacher_forcing_ratio
#             top1 = output.data.max(1)[1]
#             output = Variable(trg.data[t] if is_teacher else top1).cuda()
#         return outputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        # 初始化encoder和decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        input_length = source.size(0)  # 获取输入的长度
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).cuda()
        # 为语句中的每一个word编码
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])
        # 使用encoder的hidden层作为decoder的hidden层
        decoder_hidden = encoder_hidden.cuda()
        # 在预测前，添加一个token
        decoder_input = torch.tensor([SOS_token]).cuda()
        # 获取list中的top_k
        # 根据当前的target，预测output word
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token):
                break
        return outputs