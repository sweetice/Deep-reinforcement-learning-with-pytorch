import tkinter as tk
import numpy as np


class CrossRoadGridWorld(tk.Tk):
    def __init__(self, size):
        super(CrossRoadGridWorld, self).__init__()
        if size[0]<=4 or size[1]<=4:
            raise ValueError("Size error, the grid size must larger than 4*4")
        self.size = size
        self.gridSize = 40
        self.length = size[0]*self.gridSize
        self.width = size[1]*self.gridSize
        self.action_space =['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        #self.window = tk.Tk()
        self.title('Gridworld')
        self.geometry("{0}x{1}".format(self.length,self.width))
        self.build_grid()
        self.mainloop()
        

    def build_grid(self):

        self.canvas = tk.Canvas(self, bg='white', height= self.length, width=self.width)

        #plot the full horizon line
        for y_position in np.linspace(self.width/2-self.gridSize, self.width/2+self.gridSize, 3):
            x0, y0, x1, y1 = 0, y_position, self.width, y_position
            self.canvas.create_line(x0, y0, x1, y1)
        #plot the full vertical line
        for x_postion in np.linspace(self.length/2-self.gridSize, self.length/2+self.gridSize, 3):
            x0, y0, x1, y1 = x_postion, 0, x_postion, self.length
            self.canvas.create_line(x0, y0, x1, y1)
        #plot the short horizon line
        for y_position in range(0, self.width, self.gridSize):
            x0, y0, x1, y1 = self.width/2-self.gridSize, y_position, self.width/2+self.gridSize, y_position
            self.canvas.create_line(x0, y0, x1, y1)
        #plot the short vertical line
        for x_postion in range(0, self.length, self.gridSize):
            x0, y0, x1, y1 = x_postion, self.length/2-self.gridSize, x_postion, self.length/2+self.gridSize
            self.canvas.create_line(x0, y0, x1, y1)
            
        self.canvas.pack()
        print("The CrossRoadGrid has been established")

class  OctothorpeGridWorld(tk.Tk):

    def __init__(self, size):
        super(OctothorpeGridWorld, self).__init__()
        if size[0]<=4 or size[1]<=4:
            raise ValueError("Size error, the grid size must larger than 4*4")
        if size[0] %3 !=0 or size[1]%3 !=0:
            raise ValueError("Size error, The size of the grid world must be divisible by 3")
        self.size = size
        self.gridSize = 30
        self.length = size[0]*self.gridSize
        self.width = size[1]*self.gridSize
        self.action_space =['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        #self.window = tk.Tk()
        self.title('Octothorpegrid')
        self.geometry("{0}x{1}".format(self.length,self.width))
        self.build_grid()
        self.mainloop()
        

    def build_grid(self):

        self.canvas = tk.Canvas(self, bg='white', height= self.length, width=self.width)

        #plot the full horizon line
        for y_position in np.linspace(self.width/3-self.gridSize, self.width/3+self.gridSize, 3):
            x0, y0, x1, y1 = 0, y_position, self.width, y_position
            self.canvas.create_line(x0, y0, x1, y1)

        for y_position in np.linspace(self.width/3*2-self.gridSize, self.width/3*2+self.gridSize, 3):
            x0, y0, x1, y1 = 0, y_position, self.width, y_position
            self.canvas.create_line(x0, y0, x1, y1)

        #plot the full vertical line
        for x_postion in np.linspace(self.length/3-self.gridSize, self.length/3+self.gridSize, 3):
            x0, y0, x1, y1 = x_postion, 0, x_postion, self.length
            self.canvas.create_line(x0, y0, x1, y1)

        for x_postion in np.linspace(self.length/3*2-self.gridSize, self.length/3*2+self.gridSize, 3):
            x0, y0, x1, y1 = x_postion, 0, x_postion, self.length
            self.canvas.create_line(x0, y0, x1, y1)

        #plot the short horizon line
        for y_position in range(0, self.width, self.gridSize):
            x0, y0, x1, y1 = self.width/3-self.gridSize, y_position, self.width/3+self.gridSize, y_position
            self.canvas.create_line(x0, y0, x1, y1)
        for y_position in range(0, self.width, self.gridSize):
            x0, y0, x1, y1 = self.width/3*2-self.gridSize, y_position, self.width/3*2+self.gridSize, y_position
            self.canvas.create_line(x0, y0, x1, y1)
        #plot the short vertical line
        for x_postion in range(0, self.length, self.gridSize):
            x0, y0, x1, y1 = x_postion, self.length/3-self.gridSize, x_postion, self.length/3+self.gridSize
            self.canvas.create_line(x0, y0, x1, y1)
        for x_postion in range(0, self.length, self.gridSize):
            x0, y0, x1, y1 = x_postion, self.length/3*2-self.gridSize, x_postion, self.length/3*2+self.gridSize
            self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.pack()
        print("The Gridworld has been established")

class FullGridWorld(tk.Tk):
    def __init__(self, size):
        super(FullGridWorld, self).__init__()
        if size[0]<=4 or size[1]<=4:
            raise ValueError("Size error, the grid size must larger than 4*4")
        self.size = size
        self.gridSize = 40
        self.length = size[0]*self.gridSize
        self.width = size[1]*self.gridSize
        self.action_space =['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        #self.window = tk.Tk()
        self.title('Gridworld')
        self.geometry("{0}x{1}".format(self.length,self.width))
        self.build_grid()
        self.mainloop()
        

    def build_grid(self):

        self.canvas = tk.Canvas(self, bg='white', height= self.length, width=self.width)

        #plot the horizon line
        for y_position in range(0, self.width, 40):
            x0, y0, x1, y1 = 0, y_position, self.width, y_position
            self.canvas.create_line(x0, y0, x1, y1)
        #plot the vertical line
        for x_postion in range(0, self.length, 40):
            x0, y0, x1, y1 = x_postion, 0, x_postion, self.length
            self.canvas.create_line(x0, y0, x1, y1)

        self.canvas.pack()
        print("build signal")
