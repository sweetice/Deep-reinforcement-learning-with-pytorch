import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.autograd import grad
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter


'''
Implementation of soft actor critic
Original paper: https://arxiv.org/abs/1801.01290
Not the official implementation
'''

device = ['cuda' if torch.cuda.is_avaliable() else 'cpu']
parser = argparse.ArgumentParser()


parser.add_argument("--env_name", default="Pendulum-v0")  # OpenAI gym environment name
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)


parser.add_argument('--learning_rate', default=3e-4, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discount gamma
parser.add_argument('--capacity', default=1e6, type=int) # replay buffer size
parser.add_argument('--iteration', default=1e4, type=int) #  num of  games
parser.add_argument('--batch_size', default=64, type=int) # mini batch size

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)

args = parser.parse_args()

env = gym.make(args.env_name)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, 1)
        self.sigma_head = nn.Linear(256, 1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return mu, sigma




class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SAC():
    def __init__(self):
        super(SAC, self).__init__()

        self.policy_net = Actor(state_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Q_net = Q(state_dim, action_dim).to(device)

        self.replay_buffer = [Transition] * args.capacity
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=args.learning_rate)

        self.num_transition = 0 # pointer of replay buffer

    def forward(self, state):
        state.to(device)
        mu, sigma = self.policy_net(state)
        action = self.select_action(mu, sigma)

        v = self.value_net(state)

        return action, v

    def select_action(self, mu, sigma):
        dist = Normal(mu, sigma)
        action = dist.sample()
        return action.item() # return a scalar

    def store(self, s, a, r, s_):
        index = self.num_transition % args.capacity
        transition = Transition(s, a, r, s_)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def get_action_log_prob(self, state):
        '''
        :param state:
        :return: -1 * 1 action log prob
        '''
        batch_mu, batch_sigma = self.policy_net(state)
        dist = Normal(batch_mu, batch_sigma)
        sample_action = dist.sample()
        a_log_prob = dist.log_prob(sample_action).reshape(-1, 1)
        return a_log_prob, sample_action, dist

    def update(self):

        s = torch.tensor([t.s for t in self.replay_buffer ]).to(device)
        a = torch.tensor([t.a for t in self.replay_buffer]).to(device)
        r = torch.tensor([t.r for t in self.replay_buffer]).to(device)
        s_ = torch.tensor([t.s_ for t in self.replay_buffer]).to(device)

        for _ in range(args.gradient_step):
            for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
                bn_s = s[index]
                bn_a = a[index]
                bn_r = r[index]
                bn_s_ = s_[index]

                V = self.value_net(bn_s)
                V_ = self.value_net(bn_s_)
                Q = self.Q_net(bn_s, bn_a)
                Q_hat = bn_r + args.gamma * V_

                # get noise action
                noise_mu = torch.zeros_like(bn_s)
                noise_sigma = torch.ones_like(bn_s) * 1e-3

                dist_noise = Normal(noise_mu, noise_sigma)
                noise_action = dist_noise.sample()

                log_pi, sample_action, dist = self.get_action_log_prob(bn_s)


                # !!!Note that the actions are sampled according to the current policy,
                # instead of replay buffer. (From original paper)
                V_loss = V * (V - Q + log_pi)  # J_V

                # Single Q_net this is different from original paper!!!
                Q_loss = Q * (Q - Q_hat) # J_Q

                noise_action = sample_action + noise_action
                noise_log_pi = dist.log_prob(noise_action)
                Q_noise = self.Q_net(s, noise_action)

                pi_loss = noise_log_pi + (noise_log_pi - Q) * noise_action

                # mini batch gradient descent
                self.value_optimizer.zero_grad()
                V_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                self.Q_optimizer.zero_grad()
                Q_loss.backward(retain_graph = True)
                nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
                self.Q_optimizer.step()

                self.policy_optimizer.zero_grad()
                pi_loss.backward(retain_graph = True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()



    def save(self):
        torch.save(self..state_dict(), './PPO_gradient_penalty/net_param/actor_net.pth')
        torch.save(self.critic_net.state_dict(), './PPO_gradient_penalty/net_param/critic_net.pth')


def main():
    agent = SAC()
    state = env.reset()
    print("====================================")
    print("Collection Expreience...")
    print("====================================")

    for i in range(args.iteration):
        for _ in count():
            action, v = agent.forward(state)
            next_state, reward, done, info = env.step([action])
            agent.store(state, action, reward, next_state)

            if done:
                break
            state = next_state

        # for each gradient step
        if agent.num_transition >= args.capacity:
            agent.update()
