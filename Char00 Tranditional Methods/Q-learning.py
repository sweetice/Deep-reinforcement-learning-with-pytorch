import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

ALPHA = 0.1
GAMMA = 0.95
EPSILION = 0.9
N_STATE = 20
ACTIONS = ['left', 'right']
MAX_EPISODES = 200
FRESH_TIME = 0.1

def build_q_table(n_state, actions):
    q_table = pd.DataFrame(
    np.zeros((n_state, len(actions))),
    np.arange(n_state),
    actions
    )
    return q_table

def choose_action(state, q_table):
    #epslion - greedy policy
    state_action = q_table.loc[state,:]
    if np.random.uniform()>EPSILION or (state_action==0).all():
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_action.idxmax()
    return action_name

def get_env_feedback(state, action):
    if action=='right':
        if state == N_STATE-2:
            next_state = 'terminal'
            reward = 1
        else:
            next_state = state+1
            reward = -0.5
    else:
        if state == 0:
            next_state = 0
            
        else:
            next_state = state-1
        reward = -0.5
    return next_state, reward

def update_env(state,episode, step_counter):
    env = ['-'] *(N_STATE-1)+['T']
    if state =='terminal':
        print("Episode {}, the total step is {}".format(episode+1, step_counter))
        final_env = ['-'] *(N_STATE-1)+['T']
        return True, step_counter
    else:
        env[state]='*'
        env = ''.join(env)
        print(env)
        time.sleep(FRESH_TIME)
        return False, step_counter
        
    
def q_learning():
    q_table = build_q_table(N_STATE, ACTIONS)
    step_counter_times = []
    for episode in range(MAX_EPISODES):
        state = 0
        is_terminal = False
        step_counter = 0
        update_env(state, episode, step_counter)
        while not is_terminal:
            action = choose_action(state,q_table)
            next_state, reward = get_env_feedback(state, action)
            next_q = q_table.loc[state, action]
            if next_state == 'terminal':
                is_terminal = True
                q_target = reward
            else:
                delta = reward + GAMMA*q_table.iloc[next_state,:].max()-q_table.loc[state, action]
                q_table.loc[state, action] += ALPHA*delta
            state = next_state
            is_terminal,steps = update_env(state, episode, step_counter+1)
            step_counter+=1
            if is_terminal:
                step_counter_times.append(steps)
                
    return q_table, step_counter_times

def main():
    q_table, step_counter_times= q_learning()
    print("Q table\n{}\n".format(q_table))
    print('end')
    
    plt.plot(step_counter_times,'g-')
    plt.ylabel("steps")
    plt.show()
    print("The step_counter_times is {}".format(step_counter_times))

main() 
