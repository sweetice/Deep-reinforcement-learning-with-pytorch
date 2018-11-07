This repository will implement the classic deep reinforcement learning algorithms. The aim of this repository is to provide clear code for people to learn the deep reinforcement learning algorithm. 

In the future, more algorithms will be added and the existing codes will also be maintained.

## DQN

包含两个实现, SARSA实现和Q-learning实现。

SARSA是on-policy实现，因为没有历史经验。

Q-learning是off-policy实现，因为使用了历史经验。

## Policy Gradient

使用下面的命令可以运行已经保存好的模型


```
python Run_Model.py
```


使用下面的命令开始训练


```
pytorch_MountainCar-v0.py
```



> policyNet.pkl

这个是已经保存好的模型

## Alphago zero 

我将在兵棋游戏中复现alphago zero

欢迎关注中科院自动化所智能系统与工程中心

## Actor-Critic

这是一个算法框架，Actor-Critic下面存放的是经典的REINFORCE 方法。
## 

## TO DO
- DDPG
- ACER
- A2C
- DPPO
