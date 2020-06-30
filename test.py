
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from parl.algorithms import DQN

from Agent import Agent
from Model import Model
from PaddleEnv import PaddleEnv
from ReplayMemory import ReplayMemory

LEARN_FREQ = 10 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1000  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 64   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.9 # reward 的衰减因子，一般取 0.9 到 0.999 不等

gpu = fluid.CUDAPlace(0)
fluid.Executor(gpu)

env = PaddleEnv()
action_dim = 3 #动作 一共有三种
obs_shape = [5] #观察量五种

rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.05,  # 有一定概率随机选取动作，探索
    e_greed_decrement=10e-7)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
save_path = './Model/dqn_model.ckpt'
agent.restore(save_path)

while True:  # 训练max_episode个回合，test部分不计算入episode数量
    obs = env.reset()
    episode_reward = 0
    while True:
        action = agent.predict(obs)  # 预测动作，只选最优动作
        obs, reward, done = env.step(action)
        episode_reward += reward
        if done:
            break