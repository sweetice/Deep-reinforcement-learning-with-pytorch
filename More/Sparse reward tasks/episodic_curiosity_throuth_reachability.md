# EPISODIC CURIOSITY THROUGH R EACHABILITY 论文笔记

10.04这篇文章被提交到了arxiv上，10.25 相关公众号对这篇文章进行了推送。几位老师在群里发了推送链接，我这里边读文章边做笔记，难免有疏漏和误读，请大家谅解。


## 写在最开始：文章创新点

提出了episodic memory，用来存储observation，使用一个模型（Episodic Curiosity）来评估current observation 和 previous history，当网络判断目前的状态是新的，就会产生一个bonus reward。

EC由两个部分组成，一个是embedding network，另一个是 comparator network C。
## 文章的问题

1. 对于long episode 和 sparse reward task，存储episodic memory 将产生巨大的内存开销。对于observation 是图像的任务，这种存储将变得非常难受。

2. 尽管对memory 增加了内存的限制，但这个内存的限制也会产生问题。举例：

当agent在环境中观测到了电视，电视蕴含很多连续不变的信息，agent由于EC机制，远离电视，一段时间之后，agent的memory中电视消失，之后agent又发现了电视，又折回来。这时候agent的trajectory将出现一个大循环。


## 1. Introduction 

收到如下启发：

当agent知道在换电视频道之后马上会收到一个新的observation，那么agent就不会轻易执行这个动作。获得reward只能在做了一些努力之后才能够得到。（也就是说此时的reward并不在已经探索的环境内部）。做的努力可以用环境的动作步数来衡量。

为了估计这个，我们训练了一个神经网络近似器，给定两个观察，网络将预测多久才能够将两个观测分离（注：这显然是对环境的过拟合。）

为了确定目前的环境是否是新的，我们需要对目前的已经探索过的环境保持跟踪。一个自然的想法就是使用情节记忆（episodic memory），它存储过去的实例，这样子可以轻松地使用到达近似器来评估目前的observation和过去的observation。


我们的方法如下：

一个episode内，agent开始对环境进行探索，此时是空记忆，每一次step，agent将对新的observation和episodic memory进行比较，来判断目前是否是新的（novelty），如果确实是新的，将使用更多的步骤来脱离原本的observation，此时agent将会给自己一个bonus奖励，并将目前的observation加入观测中。这个过程将一直进行，直到episode结束。

这个方法的收敛速度是ICM的两倍以上、同时对于dense reward任务的PPO算法，不会对收敛性产生巨大的影响。

## 2. Episodic Curiosity

对于sparse reward tasks， 著名的PPO算法是不太work的。我们的模型将产生一个bonus reward $b_t$，将加在r_t上，这样augmented reward将被加起来。这对于强化学习算法会有一个很好的提升——因为产生了dense reward。这种学习算法将会更稳定和更容易收敛。


### 2.1 Episodic Curiosity Module

Episodic Curiosity （EC） model 的输入是observation ，输出是bonus b。模型由参数化和非参数化模型构成。这里有两个参数化的部分：一个嵌入网络（embedding network）和一个比较器网络（comparator network）。
![网络结构](https://s1.ax1x.com/2018/10/26/iyhRCn.png))

另外两个非参数化的组成分别是episodic memory buffer M 和 reward bonus estimation function B。
完整的EC 模型下。
![完整结构图](https://s1.ax1x.com/2018/10/26/iyhLCR.png)

接下来我详细的介绍各个部分的细节。

**Embedding and Comparator networks**

这两个都是用来估计从一个观测o_J到另一个观测o_i的n步内可达性（with-k-step-reachability），这两个小网络一起构成一个大的
R网络。






