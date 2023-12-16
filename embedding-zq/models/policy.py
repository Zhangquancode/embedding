from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, embedding_dim, embedding_layer, Gamma, dropout):
        super(Policy, self).__init__()
        # # input--embedding1
        # self.in_to_y1 = nn.Linear(4*embedding_dim, 2*embedding_dim)
        # # 初始化权重w
        # self.in_to_y1.weight.data.normal_(0, 0.01)
        # # embedding1--embedding2
        # self.y1_to_y2 = nn.Linear(2*embedding_dim, embedding_dim)
        # # 初始化权重w
        # self.y1_to_y2.weight.data.normal_(0, 0.01)
        # # embedding2--output
        # self.out = nn.Linear(embedding_dim, embedding_layer)
        # # 初始化权重w
        # self.out.weight.data.normal_(0, 0.01)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3*embedding_dim, 8*embedding_dim),
            nn.Tanh(),
            nn.LayerNorm(8*embedding_dim),
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
        # 初始化存储reward,act,observation字典
        self.rewards, self.obs, self.acts = [], [], []
        self.probs = []
        # 存储输入输出维度
        self.embedding_dim = embedding_dim
        self.embedding_layer = embedding_layer
        self.exp = True
        self.pt = False
        self.epsilon = 0.9
        # Gamma越大越容易收敛
        self.Gamma = Gamma

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, inputstate):
        # # 前向传播(wx+b)
        # inputstate = self.in_to_y1(inputstate)
        # # 激活函数
        # inputstate = F.relu(inputstate)
        # # 前向传播(wx+b)
        # inputstate = self.y1_to_y2(inputstate)
        # # 激活函数
        # inputstate = torch.sigmoid(inputstate)
        # inputstate = self.dropout(inputstate)
        # # 前向传播(wx+b)
        # act = self.out(inputstate)
        # # act = torch.sigmoid(act)
        # # return act(-1即对行进行softmax运算)
        # return F.softmax(act, dim=-1)
        return self.fc(inputstate)


# class Policy():
#     def __init__(self, embedding_dim, embedding_layer, Gamma, LearningRate, dropout):
#         # 初始化神经网络actor
#         # self.policy = PG(embedding_dim, embedding_layer, dropout).cuda(1)
#         # 初始化存储reward,act,observation字典
#         self.rewards, self.obs, self.acts = [], [], []
#         # 存储输入输出维度
#         self.embedding_dim = embedding_dim
#         self.embedding_layer = embedding_layer
#         self.exp = True
#         self.pt = False
#         self.epsilon = 0.9
#         # Gamma越大越容易收敛
#         self.Gamma = Gamma
#         # 学习率
#         self.LearningRate = LearningRate
#         # network优化器
#         # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LearningRate)

    '''第二步 定义选择动作函数'''

    def choose(self, inputstate):
        # 将ndarray类型转化为tensor类型
        # inputstate = torch.FloatTensor(inputstate)
        # 得到network的act概率，detach()--不再有梯度,numpy()--转化为ndarray类型
        # probs = self.policy(inputstate).cpu().detach().numpy()
        probs = self(inputstate).detach().cpu().numpy()
        # self.probs.append(probs)
        action = torch.zeros(len(probs), len(probs[0]))
        i = 0
        # 根据概率随机选择其中一个动作
        for batch_probs in probs:
            # if (self.pt == True):
            #     print(batch_probs)
            if (self.exp == False):
                batch_action = batch_probs.argmax()
            else:
                # if np.random.uniform() < self.epsilon:
                #     batch_action = np.random.choice(np.arange(self.embedding_layer), p=batch_probs)
                # else:
                #     batch_action = np.random.randint(0, self.embedding_layer)
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

    def store_reward(self, y_hat, y):
        # 存储一个batch的reward
        # answer.unsqueeze(0)
        # select.unsqueeze(0)
        # torch.max(y_hat, 1)[1]
        # probs = np.stack(self.probs, axis=1)
        for select, answer in zip(y_hat, y):
            # max_indices = np.argmax(prob[:length], axis=1)
            # column_counts = np.bincount(max_indices, minlength=self.embedding_layer)
            ls = F.cross_entropy(select.unsqueeze(0), answer.unsqueeze(0))
            # if np.any(column_counts > 2 * length / self.embedding_layer):
            #     punish = -np.ptp(column_counts)
            # else:
            #     punish = length / self.embedding_layer
            if torch.max(select, 0)[1] - answer == 0:
                self.rewards.append(2 - ls)
            else:
                self.rewards.append(- ls)

        # for select, answer in zip(y_hat, y):
        #     if select-answer == 0:
        #         self.rewards.append(1)
        #     else:
        #         self.rewards.append(-1)

    '''第四步 学习'''

    def learn(self, x_len):
        batch_loss = 0
        total_reward = sum(self.rewards)
        # 把一次游戏的状态、动作、奖励三个列表转为tensor
        state_tensor = torch.FloatTensor(self.obs).cuda()
        action_tensor = torch.LongTensor(self.acts).cuda()
        state_tensor = torch.transpose(state_tensor, dim0=1, dim1=0)
        action_tensor = torch.transpose(action_tensor, dim0=1, dim1=0)
        for time, step in zip(range(0, len(self.rewards)), x_len):
            # 初始化一个大小与reward相同的字典
            discounted_ep_r = np.zeros(step)
            running_add = self.rewards[time]
            # reversed反向遍历每次reward
            for t in reversed(range(0, step)):
            # for t in range(0, step):
                # 根据权重重新给每次act的reward(解决reawrd delay问题)
                discounted_ep_r[t] = running_add
                running_add = running_add * self.Gamma
                # 例如，discounted_ep_r是1*87的列表，列表的第一个值为58，最后一个值为1
            # 先减去平均数再除以标准差，就可对奖励归一化，例如，奖励列表的中间段为0，最左为+2.1，最右为-1.9.
            # discounted_ep_r -= np.mean(discounted_ep_r)
            # discounted_ep_r /= np.std(discounted_ep_r)
            reward_tensor = torch.FloatTensor(discounted_ep_r).cuda()
            # 我们可以用G值直接进行学习，但一般来说，对数据进行归一化处理后，训练效果会更好
            # log_prob = torch.log(self.policy(state_tensor[time]))  # log_prob是拥有两个动作概率的张量，一个左动作概率，一个右动作概率
            log_prob = torch.log(self(state_tensor[time][:step]))
            selected_log_probs = reward_tensor * (log_prob*action_tensor[time][:step]).sum(dim=1)  # np.arange(len(action_tensor))是log_prob的索引，
            # action_tensor由0、1组成，于是log_prob[np.arange(len(action_tensor)), action_tensor]就可以取到我们已经选择了的动作的概率，是拥有一个动作概率的张量
            batch_loss += selected_log_probs.mean()
        loss = -batch_loss/len(self.rewards)
        # # 初始化反向传播坡度(否则是每次坡度求和，所以需要初始化)
        # self.optimizer.zero_grad()
        # # 反向传播
        # loss.backward()
        # # 优化更新模型
        # self.optimizer.step()
        # 清除当前游戏字典
        self.obs, self.acts, self.rewards = [], [], []
        # self.probs = []

        return loss, total_reward

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     # parser.add_argument('--learning_rate', type=float, default=0.001)
    #     parser.add_argument('--learning_rate_policy', type=float, default=0.0008)
    #     parser.add_argument('--Gamma', type=float, default=0.99)
    #     return parser