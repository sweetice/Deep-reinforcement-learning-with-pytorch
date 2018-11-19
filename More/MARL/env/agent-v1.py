import numpy as np
import tkinter as tk
import warnings
import sys
sys.path.append('..traffic')
from traffic import CrossRoadGridWorld
warnings.filterwarnings("ignore")

class Agent():
    """docstring for Agent"""
    def __init__(self, state =[6,6]):
        super(Agent, self).__init__()
        self.state = state
        self.action_space = ["up", "down","left", "right"]
        self.colour_space = ["green", 'blue', 'red', 'yellow', 'black']
        self.colour = np.random.choice(self.colour_space)
        self.gridSize = 40
        self.world = CrossRoadGridWorld((14, 14))

        if not self.is_in_cross_road(self.state):
            raise ValueError("StateError, the agent is not in the crossroad ")
        self.build_grid()

    def is_in_cross_road(self, state):

        x, y = state[0], state[1]
        if (x in [6,7] or y in [6,7]) and x<=13 and y<=13:
            return True
        else:
            return False

    def move(self, state, action):
        x, y = state[0], state[1]
        if action not in self.action_space:
            raise ValueError("The action not in the action space")
        if action == "up" :
            y += 1 
        if action == "down":
            y -= 1
        if action == "left":
            x -= 1
        if action == "right":
            x += 1

        next_state = [x,y]
        if is_in_cross_road([x,y]):
            next_state = [x,y]
        else:
            next_state = 'out'
        return next_state

    def build_grid(self):

        x0, x1 = self.state[0]*self.gridSize, (self.state[0]+1)*self.gridSize
        y0, y1 = self.state[1] * self.gridSize, (self.state[1] + 1) * self.gridSize
        self.world.canvas.create_oval(x0, y0, x1, y1, fill=self.colour)
        self.world.canvas.pack()

    def show(self):
        self.world.mainloop()

def show(canvas, agent):
    x0, x1 = agent.state[0] * agent.gridSize, (agent.state[0] + 1) * agent.gridSize
    y0, y1 = agent.state[1] * agent.gridSize, (agent.state[1] + 1) * agent.gridSize
    canvas.create_oval(x0, y0, x1, y1, fill=agent.colour)

agent1 = Agent((6,6))
agent2 = Agent((6,7))
agent3 = Agent((6,12))
canvas = agent1.world.canvas
show(canvas, agent1)
show(canvas, agent2)
show(canvas, agent3)
canvas.mainloop()
