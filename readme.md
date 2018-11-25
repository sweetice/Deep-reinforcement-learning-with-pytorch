This repository will implement the classic deep reinforcement learning algorithms. The aim of this repository is to provide clear code for people to learn the deep reinforcement learning algorithm. 

In the future, more algorithms will be added and the existing codes will also be maintained.

![demo](figures/demo.gif)  
## Requirements

- tensorflow 1.6
- tensorboardX
- gym 0.10
- pytorch 0.4

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

Here I uploaded two DQN models, training Cartpole and mountaincar.

### Tips for MountainCar-v0

This is very sparse for MountainCar-v0, it is 0 at the beginning, only when the top of the mountain is 1, there is a reward. This leads to the fact that if the sample to the top of the mountain is not taken during training, basically the train will not come out. So you can change the reward, for example, to change to the current position of the Car is positively related. Of course, there is a more advanced approach to inverse reinforcement learning (using GAN).

![value_loss](Char1%20DQN/DQN/pic/value_loss.jpg)   
![step](Char1%20DQN/DQN/pic/finish_episode.jpg) 
This is value loss for DQN, We can see that the loss increaded to 1e13 however, the network work well. This is because the training is going on, the target_net and act_net are very different, so the calculated loss becomes very large. The previous loss was small because the reward was very sparse, resulting in a small update of the two networks.


## Policy Gradient

Use the following command to run a saved model


```
python Run_Model.py
```


Use the following command to train model


```
pytorch_MountainCar-v0.py
```



> policyNet.pkl

This is a model that I have trained.


## Actor-Critic

This is an algorithmic framework, and the classic REINFORCE method is stored under Actor-Critic.

## PPO

Proximal-Policy-Optimization

## A2C

Advantage Policy Gradient, an paper in 2017 pointed out that the difference in performance between A2C and A3C is not obvious.

## A3C

A common reproduction 

## Alphago zero 

I will reproduce AlphagoZero in wargame.

### Timeline

- 2018/11/01 gives the Resnet15Dense1 version, which has poor convergence, but the winning rate can reach 85%.
- 2018/11/12 Gives the Resnet12Dense3 version, which is the first convergence and stable trend, with a winning percentage of 92%.
- 2018/11/13 The Resnet12Dense3-v2 version is given. This version is very stable and the winning rate is gradually increasing. Performance to be observed
- 2018/11/15 After 3 days of training, Wargame AI is given - the version of Gou Chen, the hit rate is stable at over 90%.
- 2018/11/16 The rate of Chen Chen wins 96%.
- 2018/11/25 Gou Chen v2,v3 release


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
- TRPO
- DPPO

# Best RL courses
- David Silver's course [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Berkeley deep RL [link](http://rll.berkeley.edu/deeprlcourse/)
- Practical RL [link](https://github.com/yandexdataschool/Practical_RL)
- Deep Reinforcement Learning by Hung-yi Lee [link](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)
