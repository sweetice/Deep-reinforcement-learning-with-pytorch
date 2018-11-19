# coding:utf-8

import sys,numpy,os,scipy.signal,threading,time,random
import tensorflow as tf
sys.path.append('../../pythonModules')
import wgenv,common,wgfeature,wgsdata

## HELPER FUNCTIONS

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_observation(obj_main):
    wgfeature.calUpdateFeature(obj_main)
    np_mapFeature = obj_main.dic_metadata['np_mapFeature']
    np_mapFeature = numpy.swapaxes(np_mapFeature,0, 1)
    np_mapFeature = numpy.swapaxes(np_mapFeature, 1, 2)
    np_bopFeature = obj_main.dic_metadata['np_bopFeature']
    return np_mapFeature[numpy.newaxis,:,:,:],np_bopFeature[numpy.newaxis,:]

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = numpy.random.randn(*shape).astype(numpy.float32)
        out *= std / numpy.sqrt(numpy.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#
def calScore(obj_main,flag_rlaicolor):
    try:
        l_values = wgsdata.getSdataValue(obj_main.dic_metadata)
        l_scores = [l_values[0],120-l_values[2],l_values[1]] if flag_rlaicolor == 0 else [l_values[2],120-l_values[0],l_values[3]]
        return l_scores
    except Exception as e:
        print('error in test_rl.py -> calScore() : ' + str(e))
        raise
class AC_Network():
    def __init__(self, scope, trainer, nonspatial_size,map_size,dict_actionsepc):
        with tf.variable_scope(scope):
            # get size of features from action_spec and observation_spec
            self.inputs_nonspatial = tf.placeholder(shape=[None, nonspatial_size], dtype=tf.float32)
            self.inputs_spatial_screen = tf.placeholder(shape=[None, map_size[0],map_size[1],map_size[2]],dtype=tf.float32)

            # Architecture here follows Atari-net Agent described in [1] Section 4.3
            self.nonspatial_dense = tf.layers.dense(inputs=self.inputs_nonspatial,units=32,activation=tf.tanh)
            self.screen_conv1 = tf.layers.conv2d(inputs=self.inputs_spatial_screen,filters=16,kernel_size=[8,8],strides=[4,4],padding='valid',activation=tf.nn.relu)
            self.screen_conv2 = tf.layers.conv2d(inputs=self.screen_conv1,filters=32,kernel_size=[4,4],strides=[2,2],padding='valid',activation=tf.nn.relu)

            # According to [1]: "The results are concatenated and sent through a linear layer with a ReLU activation."
            screen_output_length = 1
            for dim in self.screen_conv2.get_shape().as_list()[1:]:
                screen_output_length *= dim

            self.latent_vector = tf.layers.dense(
                inputs=tf.concat([self.nonspatial_dense, tf.reshape(self.screen_conv2,shape=[-1,screen_output_length])], axis=1),units=256,activation=tf.nn.relu)

            # Output layers for policy and value estimations
            # 1 policy network for base actions
            # 16 policy networks for arguments
            #   - All modeled independently
            #   - Spatial arguments have the x and y values modeled independently as well
            # 1 value network
            self.policy_arg = dict()
            for key,value in dict_actionsepc.items():
                self.policy_arg[key] = tf.layers.dense(inputs=self.latent_vector,units=value,activation=tf.nn.softmax,kernel_initializer=normalized_columns_initializer(0.01))

            self.value = tf.layers.dense(inputs=self.latent_vector,units=1,kernel_initializer=normalized_columns_initializer(1.0))

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions_arg = dict()
                self.actions_onehot_arg = dict()
                for key, value in dict_actionsepc.items():
                    self.actions_arg[key] = tf.placeholder(shape=[None],dtype=tf.int32)
                    self.actions_onehot_arg[key] = tf.one_hot(self.actions_arg[key],value,dtype=tf.float32)

                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)


                self.responsible_outputs_arg = dict()

                for key, value in dict_actionsepc.items():
                    self.responsible_outputs_arg[key] = tf.reduce_sum(self.policy_arg[key] * self.actions_onehot_arg[key], [1])
                    # tf_c1 = tf.assign(self.responsible_outputs_arg[key],1)
                    # #shape = tf.shape(self.responsible_outputs_arg[key])
                    # shape = self.screen_conv2.get_shape()
                    # tf_c1= tf.constant(1,shape=self.screen_conv2.get_shape())
                    # self.responsible_outputs_arg[key] = tf.where(tf.equal(self.responsible_outputs_arg[key],0),tf_c1,self.responsible_outputs_arg[key])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))

                self.entropy_arg = dict()
                for key, value in dict_actionsepc.items():
                    self.entropy_arg[key] = -tf.reduce_sum(self.policy_arg[key] * tf.log(tf.clip_by_value(self.policy_arg[key],1e-20,1.)))

                self.entropy = 0
                for key, value in dict_actionsepc.items():
                    self.entropy += self.entropy_arg[key]

                self.policy_loss_arg = dict()
                for key, value in dict_actionsepc.items():
                    self.policy_loss_arg[key] = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_arg[key], 1e-20, 1.0)) * self.advantages)

                self.policy_loss = 0
                for key, value in dict_actionsepc.items():
                    self.policy_loss += self.policy_loss_arg[key]

                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                # self.gradients - gradients of loss wrt local_vars
                self.gradients = tf.gradients(self.loss,self.local_vars)
                self.var_norms = tf.global_norm(self.local_vars)
                self.grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,0.5) # old=40 2018年09月10日修改，梯度太大

                # Apply local gradients to global network
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(self.grads,self.global_vars))


## WORKER AGENT
class Worker():
    def __init__(self, name, trainer, model_path,log_path, global_episodes,flag_rlaicolor,dic_mainparas,flag_testai = False):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.flag_rlaicolor = flag_rlaicolor
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_winnums = []
        self.l_actkeys = ['remain_scores','defeat_scores','city_scores']
        self.dict_episode_scores = {'remain_scores':[],'defeat_scores':[],'city_scores':[]} #分数
        self.dict_episode_actnums = {} #动作数

        self.summary_writer = tf.summary.FileWriter(log_path)  # 修改，由外部传入
        self.flag_testai = flag_testai # 是否测试ai
        self.testai_freq = 500 # 测试ai的频率
        self.testai_runnums = 20 # 测试50场的平均胜率
        self.episode_test_winrates = [] #测试的平均胜率
        self.episode_test_winscores = [] #测试的平均净胜分
        self.prob_dec_fn = lambda x : 1.0/(0.0009*x+1.1)
        print('Initializing environment #{}...'.format(self.number))
        self.env = wgenv.ENV(dic_mainparas = dic_mainparas,flag_rlaicolor=flag_rlaicolor)
        # 初始化保存动作执行变量
        self.num_whats = self.env.getWhatNums()
        for i in range(self.num_whats):
            self.dict_episode_actnums[i] = []
        print('initlize static map feature...')
        wgfeature.cfunRegister(self.env.obj_main)
        wgfeature.calOneTimeFeature(self.env.obj_main)
        wgfeature.calUpdateFeature(self.env.obj_main)
        print('initlize static map feature done...')
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.nonspatial_size = self.env.obj_main.dic_metadata['np_bopFeature'].shape[0]
        tmp_map_size = self.env.obj_main.dic_metadata['np_mapFeature'].shape
        self.map_size = (tmp_map_size[1],tmp_map_size[2],tmp_map_size[0])
        self.l_action_keys = ['who','what','how']
        dict_actionsepc ={self.l_action_keys[0]:self.env.getWhoNums(),self.l_action_keys[1]:self.env.getWhatNums(),self.l_action_keys[2]:self.env.getHowNums()}
        self.local_AC = AC_Network(self.name, trainer, self.nonspatial_size,self.map_size,dict_actionsepc)
        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        try:
            rollout = numpy.array(rollout)
            s_np_mapFeature,s_np_bopFeature = rollout[:, 0],rollout[:,1]
            np_l_chooseArg, rewards = rollout[:,2],rollout[:,3]
            s1_np_mapFeature, s1_np_bopFeature = rollout[:,4],rollout[:,5]
            episode_end, values = rollout[:,6],rollout[:,7]

            actions_arg_stack = dict()
            for l_chooseArg in np_l_chooseArg:
                for i in range(len(self.l_action_keys)):
                    key = self.l_action_keys[i]
                    if key not in actions_arg_stack:
                        actions_arg_stack[key] = []
                    actions_arg_stack[key].append(l_chooseArg[i])

            # Here we take the rewards and values from the rollout, and use them to calculate the advantage and discounted returns
            # The advantage function uses generalized advantage estimation from [2]
            self.rewards_plus = numpy.asarray(rewards.tolist() + [bootstrap_value])
            discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
            self.value_plus = numpy.asarray(values.tolist() + [bootstrap_value])
            advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
            advantages = discount(advantages, gamma)

            # Update the global network using gradients from loss
            # Generate network statistics to periodically save
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs_spatial_screen: numpy.stack(s_np_mapFeature).reshape(-1, self.map_size[0],self.map_size[1],self.map_size[2]),
                         self.local_AC.inputs_nonspatial: numpy.stack(s_np_bopFeature).reshape(-1, self.nonspatial_size),
                         self.local_AC.advantages: advantages}

            for key,value in actions_arg_stack.items():
                feed_dict[self.local_AC.actions_arg[key]] = value

            v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                   self.local_AC.policy_loss,
                                                   self.local_AC.entropy,
                                                   self.local_AC.grad_norms,
                                                   self.local_AC.var_norms,
                                                   self.local_AC.apply_grads],
                                                  feed_dict=feed_dict)
            return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n
        except Exception as e:
            common.echosentence_color('error in test_rl.py->train():{}'.format(str(e)))
            self.env.__del__()
            raise
        except KeyboardInterrupt as k:
            common.echosentence_color('error in test_rl.py->train():{}'.format(str(k)))
            self.env.__del__()
            raise


    def getwincolor(self,dic_metadata):
        '''打印态势信息'''
        try:
            list_values = wgsdata.getSdataValue(dic_metadata)
            # cout_left = [0] * 2
            # for cur_bop in self.dic_metadata['l_rbops'] + self.dic_metadata['l_bbops'] + self.dic_metadata['l_pbops']:
            #     if cur_bop.ObjBlood > 0:
            #         cout_left[int(cur_bop.GameColor)] += 1
            # print '\t 红方剩余算子数量：%d. 剩余兵力分值：%d, 夺控得分：%d' % (cout_left[0], list_values[0], list_values[1])
            # print '\t 蓝方剩余算子数量：%d. 剩余兵力分值：%d, 夺控得分：%d' % (cout_left[1], list_values[2], list_values[3])
            return 0 if list_values[0] + list_values[1] - list_values[2] > \
                        list_values[2] + list_values[3] - list_values[0] \
                else 1
        except Exception as e:
            common.echosentence_color('wgsdata > showSDDat():{}'.format(str(e)))
            raise

    def work(self, max_episode_length, gamma, sess, coord, saver):
        try:
            episode_count = sess.run(self.global_episodes)
            total_steps = 0
            print ("Starting worker " + str(self.number))
            with sess.as_default(), sess.graph.as_default():
                t_start = time.time()
                winnums = 0
                while True:   #not coord.should_stop():
                    # Download copy of parameters from global network
                    prob = self.prob_dec_fn(episode_count) #用规则ai的概率
                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_values = []
                    # episode_frames = []
                    episode_reward = 0
                    episode_step_count = 0
                    episode_end = False

                    # 测试ai性能
                    if self.flag_testai and episode_count % self.testai_freq == 0 and episode_count != 0:
                        winrate,winscore = self.test_ai(sess,self.testai_runnums)
                        self.episode_test_winrates.append(winrate)
                        self.episode_test_winscores.append(winscore)
                        common.echosentence_color('====================','blue')
                        common.echosentence_color('测试50场的胜率为：{}% ；净胜分为 {}'.format(winrate*100,winscore))
                        common.echosentence_color('====================', 'yellow')
                        summary = tf.Summary()
                        summary.value.add(tag='test/WinRate', simple_value=float(winrate))
                        summary.value.add(tag='test/WinScore', simple_value=float(winscore))
                        self.summary_writer.add_summary(summary, episode_count)
                        self.summary_writer.flush()
                    flag_wincolor = 0
                    # Start new episode
                    flag_goon = self.env.reset()
                    episode_end = not flag_goon
                    np_mapFeature,np_bopFeature = process_observation(self.env.obj_main) #resize np_mapFeature to [1,2,0]
                    s_np_mapFeature,s_np_bopFeature = np_mapFeature,np_bopFeature
                    for i in range(self.num_whats):
                        self.dict_episode_actnums[i].append(0)

                    while not episode_end:
                        flag_baseai = False
                        if(random.random() < prob):
                            flag_goon,reward,list2_chooseArg,list_npMapFea,list_npBopFea = self.env.stepFollowBaseAI()  # 依概率选择按照规则AI走,不参与训练
                            flag_baseai = True
                        else:
                            policy_args, v = sess.run([self.local_AC.policy_arg, self.local_AC.value],
                                                      feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                                                                 self.local_AC.inputs_nonspatial: np_bopFeature})
                            # nonspatial_dense,screen_conv1,screen_conv2,latent_vector= sess.run([self.local_AC.nonspatial_dense, self.local_AC.screen_conv1,self.local_AC.screen_conv2,self.local_AC.latent_vector],
                            #                           feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                            #                                      self.local_AC.inputs_nonspatial: np_bopFeature})
                            #
                            # local_vars,global_vars = sess.run([self.local_AC.local_vars,self.local_AC.global_vars])

                            l_nppolicy = []
                            for policy_key in self.l_action_keys:
                                l_nppolicy.append(policy_args[policy_key][0, :])

                            flag_goon, reward, l_chooseArg = self.env.step(l_nppolicy[0], l_nppolicy[1], l_nppolicy[2])

                        episode_end = not flag_goon
                        np_mapFeature, np_bopFeature = process_observation(self.env.obj_main)  # resize np_mapFeature to [1,2,0]

                        if episode_end: #游戏结束，根据游戏胜负修正最后的reward
                            flag_wincolor = self.getwincolor(self.env.obj_main.dic_metadata)
                            if flag_rlaicolor == flag_wincolor:
                                reward += 200 # [0,200]

                        s1_np_mapFeature, s1_np_bopFeature = np_mapFeature, np_bopFeature
                        # if not episode_end:
                        #     s1_np_mapFeature, s1_np_bopFeature = np_mapFeature, np_bopFeature
                        # else:
                        #     s1_np_mapFeature, s1_np_bopFeature = s_np_mapFeature,s_np_bopFeature

                        # Append latest state to buffer
                        if not flag_baseai:
                            self.dict_episode_actnums[l_chooseArg[1]][-1] += 1
                            episode_buffer.append([s_np_mapFeature,s_np_bopFeature, l_chooseArg, reward, s1_np_mapFeature,s1_np_bopFeature, episode_end, v[0, 0]])
                            episode_values.append(v[0, 0])
                        else:
                            assert (len(list_npBopFea) == len(list2_chooseArg))
                            list_npMapFea = [s_np_mapFeature] + list_npMapFea  # 修复bug 动作执行之前的状态应该是s_np_mapFeature
                            list_npBopFea = [s_np_bopFeature] + list_npBopFea
                            for l_i in range(len(list2_chooseArg)):
                                l_chooseArg,s_np_mapFeature,s_np_bopFeature = list2_chooseArg[l_i],list_npMapFea[l_i],list_npBopFea[l_i]
                                s1_np_mapFeature,s1_np_bopFeature = list_npMapFea[l_i+1],list_npBopFea[l_i+1]
                                v = sess.run(self.local_AC.value,
                                             feed_dict={self.local_AC.inputs_spatial_screen: s_np_mapFeature,
                                                        self.local_AC.inputs_nonspatial: s_np_bopFeature})
                                self.dict_episode_actnums[l_chooseArg[1]][-1] += 1
                                episode_buffer.append([s_np_mapFeature, s_np_bopFeature, l_chooseArg, reward, s1_np_mapFeature,s1_np_bopFeature, episode_end, v[0, 0]])
                                episode_values.append(v[0, 0])

                        episode_reward = reward
                        s_np_mapFeature, s_np_bopFeature = s1_np_mapFeature, s1_np_bopFeature
                        total_steps += 1
                        episode_step_count += 1

                        # If the episode hasn't ended, but the experience buffer is full, then we make an update step using that experience rollout
                        if len(episode_buffer) == 30 and not episode_end and episode_step_count != max_episode_length - 1:
                            # Since we don't know what the true final return is, we "bootstrap" from our current value estimation
                            v1 = sess.run(self.local_AC.value,feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                                                            self.local_AC.inputs_nonspatial: np_bopFeature})

                            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                            episode_buffer = []
                            sess.run(self.update_local_ops)
                        if episode_end:
                            break

                    if flag_wincolor == flag_rlaicolor:
                        winnums += 1
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(numpy.mean(episode_values))
                    self.episode_winnums.append(winnums)
                    #计算分数
                    l_scores = calScore(self.env.obj_main,flag_rlaicolor)
                    for i in range(len(self.l_actkeys)):
                        self.dict_episode_scores[self.l_actkeys[i]].append(l_scores[i])

                    episode_count += 1

                    global _max_score, _running_avg_score, _episodes, _steps
                    if _max_score < episode_reward:
                        _max_score = episode_reward
                    _running_avg_score = (2.0 / 101) * (episode_reward - _running_avg_score) + _running_avg_score
                    _episodes[self.number] = episode_count
                    _steps[self.number] = total_steps

                    t_now = time.time()
                    print("{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count,
                                                                      episode_reward))
                    print("Total Steps: {}\tTotal Episodes: {}\tTotal time: {}\tMax Score: {}\tAvg Score: {}".format(numpy.sum(_steps),
                                                                                                     numpy.sum(_episodes),
                                                                                                    t_now-t_start,
                                                                                                     _max_score,
                                                                                                     _running_avg_score))
                    print(self.l_actkeys[0] + ' : ' + str(l_scores[0]) + ' '+self.l_actkeys[1] + ' : ' + str(l_scores[1]) + ' '
                          +self.l_actkeys[2] + ' : ' + str(l_scores[2]) + ' ')
                    common.echosentence_color('=='*10,'blue')
                    # Update the network using the episode buffer at the end of the episode
                    if len(episode_buffer) != 0:
                        # v1 = sess.run(self.local_AC.value,feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                        #                          self.local_AC.inputs_nonspatial: np_bopFeature})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0) # 用v1替换0; 2018年09月11日重新替换为0

                    if episode_count % 5 == 0 and episode_count != 0:
                        if episode_count % 250 == 0 and self.name == 'worker_0':
                            saver.save(sess, self.model_path + '/model-' + str(0) + '.cptk')
                            print ("Saved Model")

                        mean_reward = numpy.mean(self.episode_rewards[-5:])
                        mean_length = numpy.mean(self.episode_lengths[-5:])
                        mean_value = numpy.mean(self.episode_mean_values[-5:])
                        mean_winnums = numpy.mean(self.episode_winnums[-5:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                        summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        summary.value.add(tag='Perf/winums', simple_value=float(mean_winnums))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        for i in range(self.num_whats):
                            summary.value.add(tag='ActNums/'+str(i),simple_value=float(self.dict_episode_actnums[i][-1]))
                        for actkey in self.l_actkeys:
                            summary.value.add(tag='Scores/' + actkey, simple_value=float(self.dict_episode_scores[actkey][-1]))
                        self.summary_writer.add_summary(summary, episode_count)

                        self.summary_writer.flush()
                    if self.name == 'worker_0':
                        sess.run(self.increment)
        except Exception as e:
            common.echosentence_color('error in test_rl.py->work():{}'.format(str(e)))
            self.env.__del__()
            raise
        except KeyboardInterrupt as k:
            common.echosentence_color('error in test_rl.py->work():{}'.format(str(k)))
            self.env.__del__()
            raise

    def test_ai(self,sess,test_nums):
        try:
            winrate = 0
            winscore = 0
            for gamenum in range(test_nums):
                flag_goon = self.env.reset()
                episode_end = not flag_goon
                np_mapFeature, np_bopFeature = process_observation(self.env.obj_main)  # resize np_mapFeature to [1,2,0]
                while not episode_end:
                    policy_args, v = sess.run([self.local_AC.policy_arg, self.local_AC.value],
                                              feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                                                         self.local_AC.inputs_nonspatial: np_bopFeature})
                    l_nppolicy = []
                    for policy_key in self.l_action_keys:
                        l_nppolicy.append(policy_args[policy_key][0, :])

                    flag_goon, reward, l_chooseArg = self.env.step(l_nppolicy[0], l_nppolicy[1], l_nppolicy[2])

                    episode_end = not flag_goon
                    np_mapFeature, np_bopFeature = process_observation(self.env.obj_main)  # resize np_mapFeature to [1,2,0]

                list_values = wgsdata.getSdataValue(self.env.obj_main.dic_metadata)
                red_score = 2*list_values[0] + list_values[1] - 2* list_values[2] - list_values[3]
                myscore = red_score if self.flag_rlaicolor == 0 else -red_score
                winrate = winrate + 1 if myscore > 0 else winrate
                winscore += myscore
            winrate /= test_nums
            winscore /= test_nums
            return winrate,winscore
        except Exception as e:
            common.echosentence_color('error in test_rl.py->Worker->test_ai():{}'.format(str(e)))
            self.env.__del__()
            raise
        except KeyboardInterrupt as k:
            common.echosentence_color('error in test_rl.py->Worker->test_ai():{}'.format(str(k)))
            self.env.__del__()
            raise

def defmainparas():
    try:
        num_xd, strategy_id, num_plays, num_objcutility = 0, 1, 1, 1
        import AI_QD, wgdensestranet
        dic2_rolloutaiparas = {
            'red': {'type_ai': AI_QD.AI_QD_BASE,
                    'type_stra': 'rule-base',
                    'type_stranet': wgdensestranet.StraDenseNet,
                    },
            'blue': {'type_ai': AI_QD.AI_QD_STRA,
                     'type_stra': 'random',
                     'type_stranet': wgdensestranet.StraDenseNet,
                     },
        }
        dic_mainparas = {'str_wgrootdir': '../../',
                         'str_global_flag': 'QD',
                         'flag_show': True,
                         'num_plays': num_plays,
                         'num_objcutility': num_objcutility,
                         'num_xd': num_xd,
                         'strategy_id': strategy_id,
                         'flag_action_cache': False,
                         'flag_qd_rm': True,
                         'flag_cache': True,
                         'flag_gpu': False,
                         'flag_dllnum': 0,
                         'cuda_id': 0,
                         'flag_savestate': False,
                         'dic2_aiparas': {
                             'flag_color4acai': 0,
                             'red': {'type_ai': AI_QD.AI_QD_BASE,
                                     'type_stra': 'rule-base',
                                     # type of stratree of nodes, how to select next path, [rule-base, random, net]
                                     'type_stranet': wgdensestranet.StraDenseNet,
                                     'dic2_rolloutaiparas': dic2_rolloutaiparas,
                                     'flag_candidateactions': 'rule-base'
                                     # [rule-base, stra] how to get candidate actions
                                     },
                             'blue': {'type_ai': AI_QD.AI_QD_HA,
                                      'type_stra': 'net',
                                      'type_stranet': wgdensestranet.StraDenseNet,
                                      'dic2_rolloutaiparas': dic2_rolloutaiparas,
                                      'flag_candidateactions': 'stra'
                                      },
                         },
                         'rewrad_type': 'nowscore-base',  # ['nowsocoe-base','action-base']
                         }
        return dic_mainparas
    except Exception as e:
        print('error in defmainparas() : '+str(e))
        raise

global BattleMode
BattleMode = '1v1'
global flag_ignoreEnemyReward
flag_ignoreEnemyReward = True
if __name__ == '__main__':
    try:
        '''注： 需要修改参数说明
            修改rl颜色： flag_rlaicolor
                    defmainparas()中的dic_mainparas中dic2_paras的'red'和'blue'互换
            修改算子数： BattleMode (1v0,1v1,6v6)
            是否加载已训练模型： load_model
        注意：每次更换实验条件之后，记得修改exp_version,免得覆盖之前的游戏版本
        '''
        load_model = False
        flag_rlaicolor = 1
        exp_version = 'v0'

        flag_testai = True
        max_episode_length = 300
        gamma = .99  # Discount rate for advantage estimation and reward discounting

        str_ai = 'red' if flag_rlaicolor == 0 else 'blue'
        save_path = '../../../a3c-explog/'+ BattleMode+'_'+str_ai+'_'+exp_version+'/'
        model_path = save_path + 'model/'
        log_path = save_path + 'train_0/'
        dic_mainparas = defmainparas()

        tf.reset_default_graph()
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        common.echosentence_color('BattleMode = ' + BattleMode ,'yellow')
        print('Initializing temporary environment to retrive action_spec...')
        env = wgenv.ENV(dic_mainparas = dic_mainparas,flag_rlaicolor=flag_rlaicolor)
        wgfeature.cfunRegister(env.obj_main)

        wgfeature.calOneTimeFeature(env.obj_main)
        wgfeature.calUpdateFeature(env.obj_main)

        nonspatial_size = env.obj_main.dic_metadata['np_bopFeature'].shape[0]
        tmp_map_size = env.obj_main.dic_metadata['np_mapFeature'].shape
        map_size = (tmp_map_size[1], tmp_map_size[2], tmp_map_size[0])
        l_action_keys = ['who', 'what', 'how']
        dict_actionsepc = {l_action_keys[0]: env.getWhoNums(), l_action_keys[1]: env.getWhatNums(),l_action_keys[2]: env.getHowNums()}
        print('Initializing temporary environment to retrive action_spec done...')

        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = AC_Network('global', None, nonspatial_size,map_size,dict_actionsepc)  # Generate global network
            # num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
            num_workers = 1  # psutil.cpu_count() # Set workers to number of available CPU threads
            global _max_score, _running_avg_score, _steps, _episodes
            _max_score = -200
            _running_avg_score = 0
            _steps = numpy.zeros(num_workers)
            _episodes = numpy.zeros(num_workers)
            workers = []
            # Create worker classes
            for i in range(num_workers):
                workers.append(Worker(i, trainer, model_path, log_path, global_episodes, flag_rlaicolor,dic_mainparas,flag_testai=flag_testai))
            saver = tf.train.Saver(max_to_keep=5)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            # coord = tf.train.Coordinator()
            if load_model == True:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            # This is where the asynchronous magic happens
            # Start the "work" process for each worker in a separate thread
            while(True):
                for worker in workers:
                    worker.work(max_episode_length, gamma, sess, None, saver)
            # worker_threads = []
            # for worker in workers:
            #     worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            #     t = threading.Thread(target=(worker_work))
            #     t.start()
            #     time.sleep(0.5)
            #     worker_threads.append(t)
            # coord.join(worker_threads)
    except Exception as e:
        common.echosentence_color('error in test_rl.py-> main() : {}'.format(str(e)))
        env.__del__()
        raise
    except KeyboardInterrupt as k:
        common.echosentence_color('error in test_rl.py-> main() : {}'.format(str(k)))
        env.__del__()
        raise


