# MountainCar V0

import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import adam
from torch.distributions import Categorical

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

torch.manual_seed(1)
plt.ion()


#Hyperparameters
learning_rate = 0.02
gamma = 0.995
episodes = 1000

eps = np.finfo(np.float32).eps.item()

action_space = env.action_space.n
state_space = env.observation_space.shape[0]


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(state_space, 20)
        #self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(20, action_space)

        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

policy = Policy()
optimizer = adam.Adam(policy.parameters(), lr=learning_rate)

def selct_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    c = Categorical(probs)
    action = c.sample()


    policy.saved_log_probs.append(c.log_prob(action))
    action = action.item()
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []

    for r in policy.rewards[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Formalize reward
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + eps)

    # get loss
    for reward, log_prob in zip(rewards, policy.saved_log_probs):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()



    del policy.rewards[:]
    del policy.saved_log_probs[:]

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)
    path =  './PG_MountainCar-v0/'+'RunTime'+str(RunTime)+'.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)



def main():

    running_reward = 0
    steps = []
    for episode in count(60000):
        state = env.reset()

        for t in range(10000):
            action = selct_action(state)
            state, reward ,done, info = env.step(action)
            env.render()
            policy.rewards.append(reward)

            if done:
                print("Episode {}, live time = {}".format(episode, t))
                steps.append(t)
                plot(steps)
                break
        if episode % 50 == 0:
            torch.save(policy, 'policyNet.pkl')

        running_reward = running_reward * policy.gamma - t*0.01
        finish_episode()

if __name__ == '__main__':
    main()
