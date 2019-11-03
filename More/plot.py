import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
sns.set(style='darkgrid')

def get_info(filename):
    filename = filename.replace('.npy', '') # remove .npy
    algo, env, seed = re.split('_', filename)
    seed = int(seed)
    return algo, env, seed


def get_file_name(path='./'):
    file_names = []
    for _, __, file_name in os.walk(path):
        file_names += file_name
    data_name = [f for f in file_names if '.npy' in f]
    return data_name

def exact_data(file_name, steps):
    '''
    exact data from single .npy file
    :param file_name:
    :return: a Dataframe include time, seed, algo_name, avg_reward
    '''
    avg_reward = np.load(file_name).reshape(-1, 1)
    algo, env_name, seed = get_info(file_name)
    df = pd.DataFrame(avg_reward)
    df.columns = ['Average Return']
    df['Time Steps (1e6)'] = steps
    df['Algorithm'] = algo
    df['env'] = env_name
    df['seed'] = seed
    return df


if __name__ == '__main__':
    file_names = get_file_name('./')
    _, env_name, __ = get_info(file_names[0])
    df = pd.DataFrame([])
    steps = np.linspace(0, 1, 201)
    for file in file_names:
        data = exact_data(file, steps)
        df = pd.concat([df, data], axis=0)
    sns.lineplot(x='Time Steps (1e6)', y='Average Return', data=df, hue='Algorithm',ci=90)
    plt.title(env_name)
    plt.savefig(env_name + '.svg')
    plt.show()
