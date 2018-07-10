import numpy as np
from gridworld import GridWorld

def softmax(x):
    '''Compute softmax values of array x.

    @param x the input array
    @return the softmax array
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    

def update_critic(utility_matrix, observation, new_observation, 
                   reward, alpha, gamma, done):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix
    @return the estimation error delta
    '''
    u = utility_matrix[observation[0], observation[1]]
    u_t1 = utility_matrix[new_observation[0], new_observation[1]]
    delta = reward + ((gamma * u_t1) - u)
    utility_matrix[observation[0], observation[1]] += alpha * delta
    return utility_matrix, delta

def update_actor(state_action_matrix, observation, action, delta, beta_matrix=None):
    '''Return the updated state-action matrix

    @param state_action_matrix the matrix before the update
    @param observation the state obsrved at t
    @param action taken at time t
    @param delta the estimation error returned by the critic
    @param beta_matrix a visit counter for each state-action pair
    @return the updated matrix
    '''
    col = observation[1] + (observation[0]*4)
    if beta_matrix is None: beta = 1
    else: beta = 1 / beta_matrix[action,col]
    state_action_matrix[action, col] += beta * delta
    return state_action_matrix 

def main():

    env = GridWorld(3, 4)

    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    state_action_matrix = np.random.random((4,12))
    print("State-Action Matrix:")
    print(state_action_matrix)

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3,4))
    print("Utility Matrix:")
    print(utility_matrix)

    gamma = 0.999
    alpha = 0.001 #constant step size
    beta_matrix = np.zeros((4,12))
    tot_epoch = 300000
    print_epoch = 1000

    for epoch in range(tot_epoch):
        #Reset and return the first observation
        observation = env.reset(exploring_starts=True)
        for step in range(1000):
            #Estimating the action through Softmax
            col = observation[1] + (observation[0]*4)
            action_array = state_action_matrix[:, col]
            action_distribution = softmax(action_array)
            action = np.random.choice(4, 1, p=action_distribution)
            #To enable the beta parameter, enable the libe below
            #and add beta_matrix=beta_matrix in the update actor function
            #beta_matrix[action,col] += 1 #increment the counter
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            utility_matrix, delta = update_critic(utility_matrix, observation, 
                                                  new_observation, reward, alpha, gamma, done)
            state_action_matrix = update_actor(state_action_matrix, observation, 
                                               action, delta, beta_matrix=None)
            observation = new_observation
            if done: break
            

        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(utility_matrix)
            print("")
            print("State-Action matrix after " + str(epoch+1) + " iterations:") 
            print(state_action_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility_matrix)
    print("State-Action matrix after  " + str(tot_epoch) + " iterations:")
    print(state_action_matrix)



if __name__ == "__main__":
    main()
