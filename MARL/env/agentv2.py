import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import gridworld
from copy import deepcopy

GRIDWORLDSIZE = gridworld.CrossRoadGridWorld.size

class Agent():
    """docstring for Agent"""
    def __init__(self, state =(6,0)):
        super(Agent, self).__init__()
        self.birth_state_space = [(6,0), (7,13), (0,7), (13,6)]
        self.birth_state = np.random.choice(self.birth_state_space)
        self.action_space = ["up", "down","left", "right", "stop"]
        self.colour_space = np.linspace(10,190,10)
        self.colour = np.random.choice(self.colour_space)
        self.cross_road_space = [(6,6,), (6,7), (7,6), (7,7)]
        if not self.is_in_cross_road(self.state):
            raise ValueError("StateError, the agent is not in the crossroad ")

    def is_in_cross_road(self, state):
        x, y = state[0], state[1]
        if (x in [GRIDWORLDSIZE[0]//2, GRIDWORLDSIZE[0]//2-1] \
            or y in [GRIDWORLDSIZE[1]//2, GRIDWORLDSIZE//2+1] ) :
            return True
        else:
            return False

    def get_action_space(self,state):
        if state in self.cross_road_space:
            if state == (6,6):
                action_space = deepcopy(self.action_space).remove('up')
                action_space = deepcopy(action_space).remove('right')
            if state == (6,7):
                action_space = deepcopy(self.action_space).remove('up')
                action_space = deepcopy(action_space).remove('left')
            if state == (7,6):
                action_space = deepcopy(self.action_space).remove('down')
                action_space = deepcopy(action_space).remove('right')
            if state == (7,7):
                action_space = deepcopy(self.action_space).remove('down')
                action_space = deepcopy(action_space).remove('left')
        else:
            if state[0] == 6:
                action_space = ['down', 'stop']
            if state[0] == 7:
                action_space = ['up', 'stop']
            if state[1] == 6:
                action_space = ['left', 'stop']
            if state[1] == 7:
                action_space =['right', 'stop']

        return action_space

    def get_action(self, action_space):
        return np.random.choice(action_space)


    def move(self, state, action):
        x, y = state[0], state[1]
        if action not in self.action_space:
            raise ValueError("The action not in the action space")
        if action == "up" :
            y -= 1
        if action == "down":
            y += 1
        if action == "left":
            x -= 1
        if action == "right":
            x += 1
        next_state = [x,y]
        return next_state

    def run(self):
