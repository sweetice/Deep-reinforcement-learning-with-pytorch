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
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter



# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('MountainCar-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(args.seed)
env.seed(args.seed)

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
TrainingRecord = namedtuple('TrainingRecord', ['episode', 'reward'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.action_head = nn.Linear(64, num_action)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.state_value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 5
    buffer_capacity = 8000
    batch_size = 8

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().float()
        self.critic_net = Critic().float()
        self.buffer = [None] * self.buffer_capacity
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('runs/livetime')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 4e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        action_log_prob = c.log_prob(action)

        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        index = self.counter % self.buffer_capacity
        self.buffer[index] = transition
        self.counter += 1
        return self.counter >= self.buffer_capacity

    def update(self):
        print("The agent is updateing....")
        if self.counter>=self.buffer_capacity:
            self.training_step += 1
            if self.training_step % 100 ==0:
                print('The agent has trained {} times'.format(self.training_step))
            state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
            action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
            reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
            next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
            old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

            reward = (reward - reward.mean()) / (reward.std() + 1e-10)
            with torch.no_grad():
                target_v = reward + args.gamma * self.critic_net(next_state)

            advantage = (target_v - self.critic_net(state)).detach()
            for _ in range(self.ppo_epoch):  # iteration ppo_epoch
                for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                    # epoch iteration, PPO core!!!
                    action_prob = self.actor_net(state[index])
                    c = Categorical(action_prob)
                    action_log_prob = c.log_prob(action[index])
                    ratio = torch.exp(action_log_prob - old_action_log_prob[index])

                    L1 = ratio * advantage[index]
                    L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                    action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                    self.actor_optimizer.zero_grad()
                    action_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                    self.critic_net_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                    self.critic_net_optimizer.step()
            print("Update Finished....")
        else:
            print("Buffer is less than buff capasity! ")

def main():
    agent = PPO()
    running_step = []
    for i_epoch in range(1000):

        state = env.reset()
        env.render()

        for t in count():
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            trans = Transition(state, action, action_log_prob, reward, next_state)
            env.render()
            if agent.store_transition(trans):
                if t % 2000 == 0:
                    print(" Episode {} , the Car has run {} time ".format(i_epoch, t))
            state = next_state
            if done:
                agent.update()
                running_step.append(t)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
        if i_epoch % 50 == 0:
            print("Episode {} , the step is {} ".format(i_epoch, running_step[i_epoch]))

if __name__ == '__main__':
    main()
