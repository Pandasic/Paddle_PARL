import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from parl.algorithms import DQN
import paddle
from Agent import Agent
from Model import Model
from PaddleEnv import PaddleEnv
from ReplayMemory import ReplayMemory

LEARN_FREQ = 10 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1000  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 64   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward,done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
        #for i in range(2000):
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


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
    e_greed=0.5,  # 有一定概率随机选取动作，探索
    e_greed_decrement=10e-7)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载缓存模型
save_path = './Model/dqn_model_temp.ckpt'
if os.path.exists(save_path):
    agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 50000

# 开始训练
episode = 0
eval_time = 0

while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    total_reward = 0
    for i in range(0, 50):
        reward = run_episode(env, agent, rpm)
        total_reward += reward
        #print(reward)
        episode += 1

    # test part
    #eval_reward = total_reward/50

    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}    total_reward:{}  hit/miss={}'.format(
        episode, agent.e_greed, eval_reward,total_reward,env.hit/(env.miss+env.hit)))
    eval_time += 1

    save_path = './Model/dqn_model_temp.ckpt'
    agent.save(save_path)

    if eval_time%1 ==0:
        save_path = './Model/dqn_model_%d.ckpt'%episode
        agent.save(save_path)

# 训练结束，保存模型
save_path = './Model/dqn_model.ckpt'
agent.save(save_path)