from argparse import ArgumentParser
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy

# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self, embedding_dim, embedding_layer):                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(embedding_dim, embedding_dim)                      # 设置第一个全连接层(输入层到隐藏层)
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        # self.fc2 = nn.Linear(embedding_dim, embedding_dim)                      # 设置第一个全连接层(隐藏层到隐藏层)
        # self.fc2.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(embedding_dim, embedding_layer)                    # 设置第二个全连接层(隐藏层到输出层)
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        # x = F.relu(self.fc2(x))                                                 # 连接隐藏层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = self.dropout(x)
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self, embedding_dim, embedding_layer, batch_size, LR, epsilon, gamma, target_replace_iter, memory_capacity):
                                                                                # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(embedding_dim, embedding_layer).cuda(), Net(embedding_dim, embedding_layer).cuda()
                                                                                # 利用Net创建两个神经网络: 评估网络和目标网络
        self.e_dim = embedding_dim
        self.e_layer = embedding_layer
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.start_learn = False
        self.learn_num = 64
        self.reward_counter = 0
        self.BATCH_SIZE = batch_size                                            # 样本数量
        self.LR = LR                                                            # 学习率
        self.EPSILON = epsilon                                                  # greedy policy
        self.GAMMA = gamma                                                      # reward discount
        self.TARGET_REPLACE_ITER = target_replace_iter                          # 目标网络更新频率
        self.MEMORY_CAPACITY = memory_capacity                                  # 记忆库容量
        self.memory = np.zeros((self.MEMORY_CAPACITY,self.BATCH_SIZE, embedding_dim * 2 + 1))
                                                                                # 初始化记忆库，一行代表一个transition
        self.memory_reward = np.zeros((self.MEMORY_CAPACITY,self.BATCH_SIZE))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
                                                                                # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose(self, x):                                                        # 定义动作选择函数 (x为状态)
        length = len(x)
        action = torch.zeros(length, self.e_layer).cuda()
        action_index = torch.zeros(length, 1).cuda()
        i = 0
        for x_each in x:
            if np.random.uniform() < self.EPSILON:                              # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
                actions_value = self.eval_net.forward(x_each).cpu()             # 通过对评估网络输入状态x，前向传播获得动作值
                actions_value = torch.unsqueeze(actions_value, 0)               # 在dim=0增加维数为1的维度
                action_each = torch.max(actions_value, 1)[1].data.numpy()       # 输出每一行最大值的索引，并转化为numpy ndarray形式
                action_each = action_each[0]                                    # 输出action的第一个数
            else:                                                               # 随机选择动作
                action_each = np.random.randint(0, self.e_layer)                # 这里action随机等于0或1 (N_ACTIONS = 2)
            action_index[i] = action_each
            action[i][action_each] = 1
            i += 1
        return action.cuda(), action_index.cuda()                               # 返回选择的动作 (0或1)

    def store_transtion(self, s, a, s_):                                        # 定义记忆存储函数 (这里输入为一个transition)
        s = s.detach().cpu().numpy()
        a = a.detach().cpu().numpy()
        s_ = s_.detach().cpu().numpy()
        transition = np.hstack((s, a, s_))                                      # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        self.memory[self.memory_counter, :len(transition), :] = transition      # 置入transition
        if (self.memory_counter + 1 == self.MEMORY_CAPACITY):
            self.start_learn = True
        self.memory_counter = (self.memory_counter + 1) % self.MEMORY_CAPACITY  # memory_counter自加1

    def store_reward(self, y_hat, y):
        # 存储一个batch的reward
        i = 0
        reward = np.zeros((len(y)))
        for select, answer in zip(y_hat, y):
            if select-answer == 0:
                reward[i] = 1
            else:
                reward[i] = -1
            i += 1
        while (self.reward_counter != self.memory_counter):
            self.memory_reward[self.reward_counter, :len(reward)] = reward
            self.reward_counter = (self.reward_counter + 1) % self.MEMORY_CAPACITY

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter = (self.learn_step_counter + 1) % self.TARGET_REPLACE_ITER  # 学习步数自加1
        # 抽取记忆库中的批数据
        sample_index_x = np.random.choice(self.MEMORY_CAPACITY, self.learn_num) # 在[0, 2000)内随机抽取32个数，可能会重复
        sample_index_y = np.random.choice(self.BATCH_SIZE, self.learn_num)
        b_memory_reward = self.memory_reward[sample_index_x, sample_index_y]
        b_memory = self.memory[sample_index_x, sample_index_y, :]                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :self.e_dim]).cuda()
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, self.e_dim:self.e_dim+1].astype(int)).cuda()
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory_reward).unsqueeze(1).cuda()
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -self.e_dim:]).cuda()
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.learn_num, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--LR', type=float, default=0.01)
        parser.add_argument('--epsilon', type=float, default=0.9)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--target_replace_iter', type=float, default=100)
        parser.add_argument('--memory_capacity', type=float, default=200)
        return parser