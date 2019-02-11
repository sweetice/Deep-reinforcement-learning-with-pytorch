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

### Papers Related to the DQN


  1. Playing Atari with Deep Reinforcement Learning [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb)
  2. Deep Reinforcement Learning with Double Q-learning [[arxiv]](https://arxiv.org/abs/1509.06461) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb)
  3. Dueling Network Architectures for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1511.06581) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/3.dueling%20dqn.ipynb)
  4. Prioritized Experience Replay [[arxiv]](https://arxiv.org/abs/1511.05952) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb)
  5. Noisy Networks for Exploration [[arxiv]](https://arxiv.org/abs/1706.10295) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb)
  6. A Distributional Perspective on Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1707.06887.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/6.categorical%20dqn.ipynb)
  7. Rainbow: Combining Improvements in Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1710.02298) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb)
  8. Distributional Reinforcement Learning with Quantile Regression [[arxiv]](https://arxiv.org/pdf/1710.10044.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/8.quantile%20regression%20dqn.ipynb)
  9. Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation  [[arxiv]](https://arxiv.org/abs/1604.06057) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/9.hierarchical%20dqn.ipynb)
  10. Neural Episodic Control [[arxiv]](https://arxiv.org/pdf/1703.01988.pdf) [[code]](#)

Thanks for [higgsfield](https://github.com/higgsfield)!!!

## Policy Gradient

Thanks for [Luckysneed](https://github.com/luckysneed)'s [help](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char2%20Policy%20Gradient/REINFORCE_with_Baseline.py)!!!

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

The Asynchronous Advantage Actor Critic method (A3C) has been very influential since the paper was published. The algorithm combines a few key ideas:

- An updating scheme that operates on fixed-length segments of experience (say, 20 timesteps) and uses these segments to compute estimators of the returns and advantage function.
- Architectures that share layers between the policy and value function.
- Asynchronous updates.

## A3C

Original paper: https://arxiv.org/abs/1602.01783

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
[01] [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)  
[02] [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)  
[03] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[04] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  
[05] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)  
[06] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  
[07] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)  
[08] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
[09] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)  
[10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[11] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)  
[12] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  
[13] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)  
[14] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)  

## TO DO
- [x] DDPG
- [ ] ACER
- [ ] TRPO
- [ ] SAC
- [ ] TD3


# Best RL courses
- David Silver's course [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Berkeley deep RL [link](http://rll.berkeley.edu/deeprlcourse/)
- Practical RL [link](https://github.com/yandexdataschool/Practical_RL)
- Deep Reinforcement Learning by Hung-yi Lee [link](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)  
- RL by morvanzhou [link](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)
