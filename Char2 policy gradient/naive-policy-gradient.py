import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


#Hyperparameters
learning_rate = 0.01
gamma = 0.98

num_episode = 5000
batch_size = 32


env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

def plot_durations(episode_durations):
    plt.ion()
    plt.figure(2)
    plt.clf()
    duration_t = torch.FloatTensor(episode_durations)
    plt.title('Training')
    plt.xlabel('Episodes')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.00001)

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = self.fc1(x)
        #x = F.dropout(x, 0.5)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=-1)

        return x

policy = Policy()
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)



def train():

    episode_durations = []
    #Batch_history
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    for episode in range(num_episode):
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)

        env.render()

        for t in count():
            probs = policy(state)
            c = Categorical(probs)
            action = c.sample()

            action = action.data.numpy().astype('int32')
            next_state, reward, done, info = env.step(action)
            reward = 0 if done else reward # correct the reward
            env.render()

            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done:
                episode_durations.append(t+1)
                plot_durations(episode_durations)
                break

        # update policy
        if episode >0 and episode % batch_size == 0:

            r = 0
            '''
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma +reward_pool[i]
                    reward_pool[i] = running_add
            '''
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    r = 0
                else:
                    r = r * gamma + reward_pool[i]
                    reward_pool[i] = r

            #Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            reward_pool = (reward_pool-reward_mean)/reward_std

            #gradiend desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = policy(state)
                c = Categorical(probs)

                loss = -c.log_prob(action) * reward
                loss.backward()

            optimizer.step()

            # clear the batch pool
            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

train()
