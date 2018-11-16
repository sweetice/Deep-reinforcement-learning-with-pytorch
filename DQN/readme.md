# Requirment：

- tensorflow 1.10
- pytorch 4.1
- tensorboardX
- gym

## Tips for MountainCar-v0 env:

This is very sparse for MountainCar-v0, it is 0 at the beginning, only when the top of the mountain is 1, there is a reward. This leads to the fact that if the sample to the top of the mountain is not taken during training, basically the train will not come out. So you can change the reward, for example, to change to the current position of the Car is positively related. Of course, there is a more advanced approach to inverse reinforcement learning (using GAN).
