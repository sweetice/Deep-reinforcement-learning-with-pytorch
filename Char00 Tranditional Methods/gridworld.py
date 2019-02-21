
import numpy as np

class GridWorld:

    def __init__(self, tot_row, tot_col):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        #The world is a matrix of size row x col x 2
        #The first layer contains the obstacles
        #The second layer contains the rewards
        #self.world_matrix = np.zeros((tot_row, tot_col, 2))
        self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        #self.transition_array = np.ones(self.action_space_size) / self.action_space_size
        self.reward_matrix = np.zeros((tot_row, tot_col))
        self.state_matrix = np.zeros((tot_row, tot_col))
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]

    #def setTransitionArray(self, transition_array):
        #if(transition_array.shape != self.transition_array):
            #raise ValueError('The shape of the two matrices must be the same.') 
        #self.transition_array = transition_array        

    def setTransitionMatrix(self, transition_matrix):
        '''Set the reward matrix.

        The transition matrix here is intended as a matrix which has a line
        for each action and the element of the row are the probabilities to
        executes each action when a command is given. For example:
        [[0.55, 0.25, 0.10, 0.10]
         [0.25, 0.25, 0.25, 0.25]
         [0.30, 0.20, 0.40, 0.10]
         [0.10, 0.20, 0.10, 0.60]]

        This matrix defines the transition rules for all the 4 possible actions.
        The first row corresponds to the probabilities of executing each one of
        the 4 actions when the policy orders to the robot to go UP. In this case
        the transition model says that with a probability of 0.55 the robot will
        go UP, with a probaiblity of 0.25 RIGHT, 0.10 DOWN and 0.10 LEFT.
        '''
        if(transition_matrix.shape != self.transition_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.transition_matrix = transition_matrix

    def setRewardMatrix(self, reward_matrix):
        '''Set the reward matrix.

        '''
        if(reward_matrix.shape != self.reward_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.reward_matrix = reward_matrix

    def setStateMatrix(self, state_matrix):
        '''Set the obstacles in the world.

        The input to the function is a matrix with the
        same size of the world 
        -1 for states which are not walkable.
        +1 for terminal states
         0 for all the walkable states (non terminal)
        The following matrix represents the 4x3 world
        used in the series "dissecting reinforcement learning"
        [[0,  0,  0, +1]
         [0, -1,  0, +1]
         [0,  0,  0,  0]]
        '''
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix

    def setPosition(self, index_row=None, index_col=None):
        ''' Set the position of the robot in a specific state.

        '''
        if(index_row is None or index_col is None): self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]
        else: self.position = [index_row, index_col]

    def render(self):
        ''' Print the current world in the terminal.

        O represents the robot position
        - respresent empty states.
        # represents obstacles
        * represents terminal states
        '''
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):
                if(self.position == [row, col]): row_string += u" \u25CB " # u" \u25CC "
                else:
                    if(self.state_matrix[row, col] == 0): row_string += ' - '
                    elif(self.state_matrix[row, col] == -1): row_string += ' # '
                    elif(self.state_matrix[row, col] == +1): row_string += ' * '
            row_string += '\n'
            graph += row_string 
        print(graph)            

    def reset(self, exploring_starts=False):
        ''' Set the position of the robot in the bottom left corner.

        It returns the first observation
        '''
        if exploring_starts:
            while(True):
                row = np.random.randint(0, self.world_row)
                col = np.random.randint(0, self.world_col)
                if(self.state_matrix[row, col] == 0): break
            self.position = [row, col]
        else:
            self.position = [self.world_row-1, 0]
        #reward = self.reward_matrix[self.position[0], self.position[1]]
        return self.position

    def step(self, action):
        ''' One step in the world.

        [observation, reward, done = env.step(action)]
        The robot moves one step in the world based on the action given.
        The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        @return observation the position of the robot after the step
        @return reward the reward associated with the next state
        @return done True if the state is terminal  
        '''
        if(action >= self.action_space_size): 
            raise ValueError('The action is not included in the action space.')

        #Based on the current action and the probability derived
        #from the trasition model it chooses a new actio to perform
        action = np.random.choice(4, 1, p=self.transition_matrix[int(action),:])
        #action = self.transition_model(action)

        #Generating a new position based on the current position and action
        if(action == 0): new_position = [self.position[0]-1, self.position[1]]   #UP
        elif(action == 1): new_position = [self.position[0], self.position[1]+1] #RIGHT
        elif(action == 2): new_position = [self.position[0]+1, self.position[1]] #DOWN
        elif(action == 3): new_position = [self.position[0], self.position[1]-1] #LEFT
        else: raise ValueError('The action is not included in the action space.')

        #Check if the new position is a valid position
        #print(self.state_matrix)
        if (new_position[0]>=0 and new_position[0]<self.world_row):
            if(new_position[1]>=0 and new_position[1]<self.world_col):
                if(self.state_matrix[new_position[0], new_position[1]] != -1):
                    self.position = new_position

        reward = self.reward_matrix[self.position[0], self.position[1]]
        #Done is True if the state is a terminal state
        done = bool(self.state_matrix[self.position[0], self.position[1]])
        return self.position, reward, done

