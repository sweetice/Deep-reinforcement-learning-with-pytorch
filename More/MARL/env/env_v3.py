import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import agent
import gridworld
from copy import deepcopy

class Env():
    def __init__(self):
        super(Env, self).__init__()
        self.CrossRoadGridWorld = gridworld.CrossRoadGridWorld()
        self.state = self.CrossRoadGridWorld.init_state
        self.max_num_agent = 10
        self.state_space = []

    def run(self):
        pass

    def update_state(self, state):
        self.state = state
        self.state_space.append(state)

    def show(self):
        fig, ax = plt.subplots()

        for i in range(len(self.state_space)):
            ax.cla()
            ax.set_title("frame {}".format(i))
            ax.imshow(self.state_space[i])
            plt.pause(0.1)
step = 20
def mian():
    agent0 = agent.Agent()
    env = Env()
    for  i in range(step):
        state = agent0.state
        action = agent0.get_action()
        agent0.state = agent0.move(state, action)
        env_state = deepcopy(env.state)
        env_state[agent0.state[0], agent0.state[1]]   = agent0.colour
        env.state_space.append(env_state)
    fig, ax = plt.subplots()
    for i in range(step):
        ax.cla()
        ax.grid()
        #ax.grid()
        ax.imshow(env.state_space[i])
        ax.set_title("frame {}".format(i+1))
        plt.pause(0.3)

if __name__ == '__main__':
    mian()
