import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class CrossRoadGridWorld():
    def __init__(self, size=(14,14)):
        super(CrossRoadGridWorld, self).__init__()
        self.state = np.zeros(size)
        self.size = size
        self.init_state = self.get_init_state()
        if size[0] <= 4 or size[1] <= 4:
            raise ValueError("Size error, the grid size must be larger than 4*4")
        self.title = "Cross Road Grid World"
        self.length = size[0]
        self.width = size[1]

    def reset(self):
        self.state = self.get_init_state()

    def get_init_state(self):
        #plot the horizon block
        init_state = np.zeros((self.size))
        for i in range(self.size[0]):
            for j in [self.size[1]//2-1, self.size[1]//2]:
                init_state[i,j] = 200

        #plot the vertical block
        for i in [self.size[0]//2-1, self.size[0]//2]:
            for j in range(self.size[1]):
                init_state[i,j] = 200

        return  init_state
