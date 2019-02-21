import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn.functional import  smooth_l1_loss

#Hyperparameters
LEARNING_RATE = 0.01
GAMMA = 0.995
NUM_EPISODES = 50000
RENDER = False
#env info
env = gym.make('MountainCar-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

num_state = env.observation_space.shape[0]
num_action = env.action_space.n
eps = np.finfo(np.float32).eps.item()
plt.ion()
saveAction = namedtuple('SavedActions',['probs', 'action_values'])

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        #self.fc2 = nn.Linear(64, 128)

        self.action_head = nn.Linear(128, num_action)
        self.value_head = nn.Linear(128, 1)
        self.policy_action_value = []
        self.rewards = []

        self.gamma = GAMMA
        os.makedirs('/AC_MountainCar-v0_Model/', exist_ok=True)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))

        probs = F.softmax(self.action_head(x))
        value = self.value_head(x)
        return probs, value

policy = Module()
optimizer = Adam(policy.parameters(), lr=LEARNING_RATE)

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)
    path = './AC_MountainCar-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 200 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, value = policy(state)
    c = Categorical(probs)
    action = c.sample()
    log_prob = c.log_prob(action)


    policy.policy_action_value.append(saveAction(log_prob, value))
    action = action.item()
    return action


def finish_episode():
    rewards = []
    saveActions = policy.policy_action_value
    policy_loss = []
    value_loss = []
    R = 0

    for r in policy.rewards[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Normalize the reward
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    #Figure out loss
    for (log_prob, value), r in zip(saveActions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
        value_loss.append(smooth_l1_loss(value, torch.tensor([r]) ))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.policy_action_value[:]


def main():
    run_steps = []
    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        if RENDER: env.render()

        for t in count():
            action = select_action(state)
            state , reward, done, _ = env.step(action)
            reward = state[0] + reward
            if RENDER: env.render()
            policy.rewards.append(reward)

            if done:
                run_steps.append(t)
                print("Epiosde {} , run step is {} ".format(i_episode+1 , t+1))
                break

        finish_episode()
        plot(run_steps)

        if i_episode % 100 == 0 and i_episode !=0:

            modelPath = './AC_MountainCar-v0_Model/ModelTraing' + str(i_episode) + 'Times.pkl'
            torch.save(policy, modelPath)



if __name__ == '__main__':
    main()
