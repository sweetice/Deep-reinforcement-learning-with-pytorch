This repository will implement the classic deep reinforcement learning algorithms. The aim of this repository is to provide clear code for people to learn the deep reinforcement learning algorithm. 

In the future, more algorithms will be added and the existing codes will also be maintained.

## Installation
1. install the pytorch
```bash
plase go to official webisite to install it: https://pytorch.org/

Recommend use Anaconda Virtual Environment to manage your packages

```
2. install openai-baselines (**the openai-baselines update so quickly, please use the older version as blow, will solve in the future.**)
```bash
# clone the openai baselines
git clone https://github.com/openai/baselines.git
cd baselines
git checkout 366f486
pip install -e .

```

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


## Actor-Critic

这是一个算法框架，Actor-Critic下面存放的是经典的REINFORCE 方法。

## PPO

Proximal-Policy-Optimization

## Alphago zero 

我将在兵棋游戏中复现alphago zero
### Timeline
- 2018/11/01 给出Resnet15Dense1 版本，该版本收敛性不好，但胜率能够达到85%.只能够找点适用
- 2018/11/12 给出Resnet12Dense3 版本，该版本第一次出现收敛、稳定趋势，胜率达到92%
- 2018/11/13 给出Resnet12Dense3-v2版本， 该版本收敛性非常稳定，胜率逐步上升。性能待观测

欢迎关注中科院自动化所智能系统与工程中心

## Papers Related to the Deep Reinforcement Learning
[1] [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)  
[2] [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)  
[3] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[4] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)  
[6] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  
[7] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)  
[8] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
[9] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)  
[10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[11] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)  

## TO DO
- DDPG
- ACER
- A2C
- DPPO
