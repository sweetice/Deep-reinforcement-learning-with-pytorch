import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import gym

#hyper parameters

EPSILON = 0.9
GAMMA = 0.9
LR = 0.01
MEMORY_CAPACITY = 200
Q_NETWORK_ITERATION = 50
BATCH_SIZE = 4


env = gym.make('MountainCar-v0')

NUM_STATES = env.state_space.n
NUM_ACTIONS = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_states = NUM_STATES
        self.num_actions = NUM_ACTIONS
        self.fc1 = nn.Linear(self.num_states, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(10, self.num_actions)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        action = self.fc2(state)

        return action

class Dqn():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory = np.zeros((MEMORY_CAPACITY, 4))
        self.memory_counter = 0
        self.learn_counter = 0
        # state, action ,reward and next state 4

    def store_trans(self, state, action, reward, next_state):
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, action, reward, next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        # EPSILON
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[0].data.numpy()
        else:
            action = np.random.choice()

