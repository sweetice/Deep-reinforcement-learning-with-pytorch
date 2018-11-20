import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def discount(sequence, Gamma = 0.99):
    R = 0
    reward = []
    for r in sequence[::-1]:
        R = r + Gamma * R
        reward.insert(0, R)
    return reward


def makedir():
    if not os.path.exists('./exp'):
        os.makedirs('./exp/model')
        os.makedirs('./exp/logs')


def save_model(model, iteration_time):
    path = './exp/model/model'+str(iteration_time)+'.pkl'
    torch.save(model.state_dict(), path)


