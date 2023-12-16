import math
import torch
import random
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu
# from torchmetrics.functional import bleu_score
# from torchtext.data.metrics import bleu_score
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

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_list,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        # 英语词典数量
        self.input_size = input_size
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 嵌入层大小
        self.embed_size = embed_size
        # 英语词典
        self.vocab_list = vocab_list
        # embedding1
        self.embedding1 = nn.Embedding(input_size, embed_size)
        self.embedding1.weight.data.uniform_(-1, 1)
        self.embedding1.weight.requires_grad = True

        # glove
        # embeddings_dict = {}
        # with open("../embedding-zq/glove/glove.6B." + str(embed_size) + "d.txt", 'r', encoding="utf-8") as f:
        # # with open("./glove/glove.6B."+str(embed_size)+"d.txt", 'r', encoding="utf-8") as f:
        #     for line in f:
        #         values = line.split()
        #         word = values[0]
        #         vector = np.asarray(values[1:], "float32")
        #         embeddings_dict[word] = vector
        self.embedding2 = nn.Embedding(input_size, embed_size)
        self.embedding2.weight.data.normal_(0, 0.1)

        # glove_vector = []
        # for word in self.vocab_list:
        #     if word in embeddings_dict:
        #         glove_vector.append(embeddings_dict[word])
        #     else:
        #         glove_vector.append(np.random.randn(embed_size))
        # glove_vector = np.array(glove_vector)
        #
        # self.embedding2.weight.data.copy_(torch.from_numpy(glove_vector))
        self.embedding2.weight.requires_grad = True #是否随着网络进行更新


        # gensim
        # embeddings_dict2 = {}
        # with open("../embedding-zq/models/w2cmodel/word2vec_" + str(embed_size) + ".txt", 'r', encoding="utf-8") as f:
        # # with open("./w2cmodel/word2vec_"+str(embed_size)+".txt", 'r', encoding="utf-8") as f:
        #     for line in f:
        #         values2 = line.split()
        #         word2 = values2[0]
        #         vector2 = np.asarray(values2[1:], "float32")
        #         embeddings_dict2[word2] = vector2
        self.embedding3 = nn.Embedding(input_size, embed_size)
        vector = np.random.randn(input_size, embed_size)
        self.embedding3.weight.data.copy_(torch.from_numpy(vector))

        # gensim_vector = []
        # for word in self.vocab_list:
        #     if word in embeddings_dict2:
        #         gensim_vector.append(embeddings_dict2[word])
        #     else:
        #         gensim_vector.append(np.random.randn(embed_size))
        # gensim_vector = np.array(gensim_vector)
        #
        # self.embedding3.weight.data.copy_(torch.from_numpy(gensim_vector))
        self.embedding3.weight.requires_grad = True #是否随着网络进行更新

        # self.rnncell = nn.GRUCell(embed_size, hidden_size)
        self.rnncell = nn.RNNCell(embed_size, hidden_size)
        # self.rnncell = nn.LSTMCell(embed_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1, inplace=True)


    def forward(self, src, policy, hidden=None):
        # 当1emb时选择哪个emb
        model = 1
        # batchsize
        batch_size = 64
        # rnntype
        rnn_type = 'RNN'#LSTM, GRU, RNN
        # 选择几个emb
        layer = policy.embedding_layer
        # dropout
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
                        h = self.relu(h)
                        output = h.unsqueeze(1)

                    elif rnn_type == 'LSTM':
                        output = h[0].unsqueeze(1)
                else:
                    # test = self.dropout(test, 0.5)
                    h = self.rnncell(test, h)
                    if rnn_type == 'RNN' or rnn_type == 'GRU':
                        h = self.relu(h)
                        output = torch.cat((output, h.unsqueeze(1)), dim=1)

                    elif rnn_type == 'LSTM':
                        output = torch.cat((output, h[0].unsqueeze(1)), dim=1)
                i += 1
        elif layer == 2:
            # 设置初始权重
            # {Tensor:(1,2)}tensor([[0.5,0.5]])
            e_index = torch.tensor([[1 / 2, 1 / 2]]).cuda()
            # 获得词向量维度
            # index = len(e1[0][0])
            # {Tensor:(200,200)}
            # {Tensor:(64,200)}
            # ???:这里的h或许是论文中的h
            # h = torch.zeros(batch_size, index)#.cuda()
            i = 0
            for test1, test2, test3 in zip(e1, e2, e3):

                merge = torch.cat((test1, test3), dim=1)
                options = torch.stack((test1, test3), dim=1)

                if i == 0:
                    # 用e_index=tensor([[0.5,0.5]])融合在一起
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
            # h = torch.zeros(batch_size, index).cuda()
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
        output = torch.transpose(output, dim0=1, dim1=0)

        if rnn_type == 'RNN' or rnn_type == 'GRU':
            return output, h.unsqueeze(0)
        elif rnn_type == 'LSTM':
            return output, h[0].unsqueeze(0)

    # def dropout(self, X, drop_prob):
    #     X = X.float().cuda()
    #     assert 0 <= drop_prob <= 1
    #     keep_prob = 1 - drop_prob
    #     if keep_prob == 0:
    #         return torch.zeros_like(X)
    #     mask = (torch.randn(X.shape) < keep_prob).float().cuda()
    #     return mask * X / keep_prob

class Policy(nn.Module):
    def __init__(self, embedding_dim, embedding_layer):
        super(Policy, self).__init__()
        # # input--embedding1
        # self.in_to_y1 = nn.Linear(4 * embedding_dim, 8 * embedding_dim)
        # # 初始化权重w
        # self.in_to_y1.weight.data.normal_(0, 0.1)
        # self.layer1 = nn.LayerNorm(8 * embedding_dim)
        # # embedding1--embedding2
        # self.y1_to_y2 = nn.Linear(8 * embedding_dim, 8 * embedding_dim)
        # # 初始化权重w
        # self.y1_to_y2.weight.data.normal_(0, 0.1)
        # self.layer2 = nn.LayerNorm(8 * embedding_dim)
        # # embedding2--embedding3
        # self.y2_to_y3 = nn.Linear(8 * embedding_dim, 4 * embedding_dim)
        # # 初始化权重w
        # self.y2_to_y3.weight.data.normal_(0, 0.1)
        # self.layer3 = nn.LayerNorm(4 * embedding_dim)
        # # embedding3--embedding4
        # self.y3_to_y4 = nn.Linear(4 * embedding_dim, 2 * embedding_dim)
        # # 初始化权重w
        # self.y3_to_y4.weight.data.normal_(0, 0.1)
        # self.layer4 = nn.LayerNorm(2 * embedding_dim)
        # # embedding4--output
        # self.out = nn.Linear(2 * embedding_dim, embedding_layer)
        # # 初始化权重w
        # self.out.weight.data.normal_(0, 0.1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim, 8 * embedding_dim),
            nn.Tanh(),
            nn.LayerNorm(8 * embedding_dim),
            torch.nn.Linear(8 * embedding_dim, 8 * embedding_dim),
            nn.Tanh(),
            nn.LayerNorm(8 * embedding_dim),
            torch.nn.Linear(8 * embedding_dim, 4 * embedding_dim),
            nn.Tanh(),
            nn.LayerNorm(4 * embedding_dim),
            torch.nn.Linear(4 * embedding_dim, 2 * embedding_dim),
            nn.Tanh(),
            nn.LayerNorm(2 * embedding_dim),
            torch.nn.Linear(2 * embedding_dim, embedding_layer),
            nn.Softmax(dim=1)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
        self.dropout = nn.Dropout(p=0.1)

        # 初始化存储reward,act,observation字典
        self.rewards, self.obs, self.acts = [], [], []
        # 存储输入输出维度
        self.embedding_dim = embedding_dim
        self.embedding_layer = embedding_layer
        self.exp = True
        # Gamma越大越容易收敛
        self.Gamma = 0.99

    def forward(self, inputstate):
        # # 前向传播(wx+b)
        # inputstate = self.in_to_y1(inputstate)
        # # 激活函数
        # inputstate = F.tanh(inputstate)
        # inputstate = self.layer1(inputstate)
        # # 前向传播(wx+b)
        # inputstate = self.y1_to_y2(inputstate)
        # # 激活函数
        # inputstate = F.tanh(inputstate)
        # inputstate = self.layer2(inputstate)
        # # inputstate = self.dropout(inputstate)
        # # 前向传播(wx+b)
        # inputstate = self.y2_to_y3(inputstate)
        # # 激活函数
        # inputstate = F.tanh(inputstate)
        # inputstate = self.layer3(inputstate)
        # # 前向传播(wx+b)
        # inputstate = self.y3_to_y4(inputstate)
        # # 激活函数
        # inputstate = F.tanh(inputstate)
        # inputstate = self.layer4(inputstate)
        # # 前向传播(wx+b)
        # act = self.out(inputstate)
        # # act = torch.sigmoid(inputstate)
        # # return act(-1即对行进行softmax运算)
        # return F.softmax(act, dim=-1)
        return self.fc(inputstate)

    '''第二步 定义选择动作函数'''

    def choose(self, inputstate):
        # 将ndarray类型转化为tensor类型
        # inputstate = torch.FloatTensor(inputstate)
        # 得到network的act概率，detach()--不再有梯度,numpy()--转化为ndarray类型
        probs = self(inputstate).detach().cpu().numpy()
        action = torch.zeros(len(probs), len(probs[0]))
        i = 0
        # 根据概率随机选择其中一个动作
        for batch_probs in probs:
            if (self.exp == False):
                batch_action = batch_probs.argmax()
            else:
                batch_action = np.random.choice(np.arange(self.embedding_layer), p=batch_probs)
            action[i][batch_action] = 1
            i += 1

        # 返回动作
        return action.cuda()

    '''第三步 存储每一个回合的数据'''

    def store_transtion(self, s, a):
        # 将每次observation, action存入字典
        s = s.detach().cpu().numpy()
        a = a.detach().cpu().numpy()
        self.obs.append(s)
        self.acts.append(a)

    def store_reward(self, y_hat, y, trg_field):

        for y_hat_s, y_s in zip(y_hat, y):
            trgs = []
            pred_trgs = []
            y_hat_s = y_hat_s.argmax(1).detach().cpu().numpy()
            y_hat_tokens = [trg_field.vocab.itos[i] for i in y_hat_s]
            y_hat_tokens = y_hat_tokens[:-1]
            # pred_trgs.append(y_hat_tokens)

            rewards = (y_s.eq(torch.tensor(y_hat_s).cuda()) + 0) * 2 - 1
            # rewards = (y_s.eq(torch.tensor(y_hat_s)) + 0) * 2 - 1

            y_s = y_s.cpu().numpy()
            y_tokens = [trg_field.vocab.itos[ii] for ii in y_s]
            y_tokens = y_tokens[:-1]
            # trgs.append([y_tokens])
            bleu = sentence_bleu([y_tokens], y_hat_tokens)
            self.rewards.append((bleu + rewards.sum()/len(rewards)))
        # outputs = y_hat.transpose(0, 1).detach()
        # trgs = y.transpose(0, 1).detach()
        # y_hat = torch.max(torch.exp(y_hat), 2)[1].transpose(0, 1)
        # y = y.transpose(0, 1)
        # rewards = (y.eq(y_hat) + 0) * 2 - 1
        # rewards[:, 0] = 0
        # # 存储一个batch的reward
        # for reward, output, trg in zip(rewards, outputs, trgs):
        #     ls = F.nll_loss(output[1:], trg[1:], ignore_index=1).detach()
        #     self.rewards.append(reward.sum() - ls)

    '''第四步 学习'''

    def learn(self, trg_len):
        batch_loss = 0
        reward = sum(self.rewards) / len(self.rewards)
        # rewards_tensor = torch.stack(self.rewards).cpu()
        # mean_value = torch.mean(rewards_tensor)#.cuda()
        # std_value = torch.std(rewards_tensor)#.cuda()
        # self.rewards = [(tensor - mean_value) / std_value for tensor in self.rewards]
        # 把一次游戏的状态、动作、奖励三个列表转为tensor
        state_tensor = torch.FloatTensor(self.obs).cuda()
        action_tensor = torch.LongTensor(self.acts).cuda()
        state_tensor = torch.transpose(state_tensor, dim0=1, dim1=0)
        action_tensor = torch.transpose(action_tensor, dim0=1, dim1=0)
        for time, step in zip(range(0, len(self.rewards)), trg_len):
            # 初始化一个大小与reward相同的字典
            discounted_ep_r = np.zeros(step-2)
            running_add = self.rewards[time]
            # reversed反向遍历每次reward
            for t in reversed(range(0, step-2)):
                # 根据权重重新给每次act的reward(解决reawrd delay问题)
                discounted_ep_r[t] = running_add
                running_add = running_add * self.Gamma
                # 例如，discounted_ep_r是1*87的列表，列表的第一个值为58，最后一个值为1
            # 先减去平均数再除以标准差，就可对奖励归一化，例如，奖励列表的中间段为0，最左为+2.1，最右为-1.9.
            # discounted_ep_r -= np.mean(discounted_ep_r)
            # if running_add > 0:
            #     discounted_ep_r /= np.std(discounted_ep_r)
            reward_tensor = torch.FloatTensor(discounted_ep_r).cuda()
            # 我们可以用G值直接进行学习，但一般来说，对数据进行归一化处理后，训练效果会更好
            log_prob = torch.log(self(state_tensor[time][1:step-1]))  # log_prob是拥有两个动作概率的张量，一个左动作概率，一个右动作概率
            selected_log_probs = reward_tensor * (log_prob * action_tensor[time][1:step-1]).sum(
                dim=1)  # np.arange(len(action_tensor))是log_prob的索引，
            # action_tensor由0、1组成，于是log_prob[np.arange(len(action_tensor)), action_tensor]就可以取到我们已经选择了的动作的概率，是拥有一个动作概率的张量
            batch_loss += selected_log_probs.mean()
        loss = -batch_loss / len(self.rewards)
        # 清除当前游戏字典
        self.obs, self.acts, self.rewards = [], [], []
        return loss, reward


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 全连接层
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        # 类似一个词向量？
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        # 初始化在1/200^(1/2)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # 源句子步数(单词数)
        timestep = encoder_outputs.size(0)
        # 重复最后的状态(变成全部状态的形状)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        # x,y转置
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        # 计算分数（大概是key-value分数）
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
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        # 嵌入层大小
        self.embed_size = embed_size
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 德语词典数量
        self.output_size = output_size
        # rnn层数
        self.n_layers = n_layers

        # 嵌入层
        self.embed = nn.Embedding(output_size, embed_size)
        # dropout
        self.dropout = nn.Dropout(dropout, inplace=True)
        # Attention(nn.Module)
        self.attention = Attention(hidden_size)
        # gru
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        # 输出层
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        # 将词向量变成三维
        embedded = self.embed(input).unsqueeze(0)
        # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs(得到attn权重)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # attn计算结果
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        # gru
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        # 输出层
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        # output输出翻译单词 hidden下一时刻状态 attn_weights注意力权重
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, policy):
        super(Seq2Seq, self).__init__()
        # encoder
        self.encoder = encoder
        # decoder
        self.decoder = decoder
        # policy
        self.policy = policy

        self.lambda1 = 1

        self.lambda2 = 0


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 得到batchsize
        batch_size = src.size(1)
        # 得到目标句子最大长度
        max_len = trg.size(0)
        # 英语词典数量
        vocab_size = self.decoder.output_size
        # 初始化一个可反向传播的tensor
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        # 总词数矩阵与最后一个词状态矩阵
        encoder_output, hidden = self.encoder(src, self.policy)
        hidden = hidden[:self.decoder.n_layers]
        # 初始化一个trg第一行的可反向传播的tensor(decoder中下方输入的标签)
        output = Variable(trg.data[0, :])  # sos
        # 输出目标句子的每个单词
        for t in range(1, max_len):
            # decoder
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            # 保持输出翻译单词
            outputs[t] = output
            # 50概率为T\F
            is_teacher = random.random() < teacher_forcing_ratio
            # 返回最大概率的位置，即选择的单词
            top1 = output.data.max(1)[1]
            # 一半概率用真实的、一半概率用训练的(decoder中下方输入的单词)
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs
