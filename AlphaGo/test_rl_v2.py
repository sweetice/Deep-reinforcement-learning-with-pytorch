# coding:utf-8

import sys,numpy,os,scipy.signal,threading,time,random,copy
import tensorflow as tf
sys.path.append('../../pythonModules')
import wgenv,common,wgfeature,wgsdata
from resnet_model import  ResNet
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

def filterMetadata(env,BattleMode):
    '''sgr修改，去除部分加载算子，在简单情况下验证算法
        battlemode = 66
    '''
    try:

        l_rbops,l_bbops, l_pbops= env.obj_main.dic_metadata['l_rbops'],env.obj_main.dic_metadata['l_bbops'],env.obj_main.dic_metadata['l_pbops']
        if BattleMode == '6v6':
            return l_rbops, l_bbops, l_pbops
        elif BattleMode == '1v0':
            l_f_rbops = []
            l_f_bbops = []
            for bop in l_bbops:
                if bop.ObjTypeX == 0:
                    l_f_bbops.append(bop)
                    break
            l_f_pbops = []
            return l_f_rbops,l_f_bbops,l_f_pbops
        elif BattleMode == '1v1':
            l_f_rbops = []
            l_f_bbops = []
            for bop in l_bbops:
                if bop.ObjTypeX == 0:
                    l_f_bbops.append(bop)
                    break
            for bop in l_rbops:
                if bop.ObjTypeX == 0:
                    l_f_rbops.append(bop)
                    break
            l_f_pbops = []
            l_rbops, l_bbops, l_pbops = l_f_rbops, l_f_bbops, l_f_pbops
    except Exception as e:
        print('error in wgsdata.py -> filterMetadata() ' + str(e))
        raise

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
    def __init__(self, scope, trainer, trainer_critic, nonspatial_size,map_size,dict_actionsepc):
        '''modify0 2018年11月07日 sgr 修改网络节点的name属性，共享节点.name =scope/public...
                ，policy = scope/policy... value = scope/value...为了反向传播时只更新特定层的梯度
        modify1 2018年11月07日 sgr 增加dic_hparas['UpdateGradsMethod'] in ['Integration','Divide'] 指示整体训练policy和value网，
                还是分开训。分训的时候value 网只更新全连接层参数
        '''
        try:
            with tf.variable_scope(scope):
                # get size of features from action_spec and observation_spec
                assert  dic_hparas['NetType'] in ['AtariNet','ResNet']
                with tf.variable_scope('public'): # name 'scope/public/...'
                    if dic_hparas['NetType'] == 'AtariNet':
                        self.inputs_nonspatial = tf.placeholder(shape=[None, nonspatial_size], dtype=tf.float32)
                        self.inputs_spatial_screen = tf.placeholder(shape=[None, map_size[0], map_size[1], map_size[2]],dtype=tf.float32)
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
                    elif dic_hparas['NetType'] == 'ResNet':
                        self.inputs_nonspatial = tf.placeholder(shape=[None, nonspatial_size], dtype=tf.float32)
                        self.inputs_spatial_screen = tf.placeholder(shape=[None, map_size[0], map_size[1], map_size[2]],dtype=tf.float32)
                        # Architecture here follows ResNet
                        batch_size = self.inputs_spatial_screen.get_shape().as_list()[0]
                        self.resnet = ResNet(self.inputs_spatial_screen,batch_size)
                        self.latent_vector = self.resnet._build_model()

                        ### 下面是HQ加的网络
                        self.policy_dense_1 = {}
                        self.policy_dense_2 = {}
                        self.value_dense_1 = None
                        self.value_dense_2 = None
                        self.policy_arg = {}
                        self.value = None

                    else:
                        common.echosentence_color('dic_hparas.NetType error.')

                # Output layers for policy and value estimations
                # 1 policy network for base actions
                # 16 policy networks for arguments
                #   - All modeled independently
                #   - Spatial arguments have the x and y values modeled independently as well
                # 1 value network
                with tf.variable_scope('policy'): # name = 'scope/policy/...'

                    for key,value in dict_actionsepc.items():
                        self.policy_dense_1[key] = tf.layers.dense(inputs=self.latent_vector,units=128,activation=tf.nn.tanh)
                        self.policy_dense_2[key] = tf.layers.dense(inputs=self.policy_dense_1[key], units=64,
                                                                   activation=tf.nn.tanh)
                        self.policy_arg[key] = tf.layers.dense(inputs=self.policy_dense_2[key], units=value,
                                                               activation=tf.nn.softmax,
                                                               kernel_initializer=normalized_columns_initializer(0.01))

                with tf.variable_scope('value'): # name = 'scope/value/...'
                    self.value_dense_1 = tf.layers.dense(inputs=self.latent_vector, units=128,
                                                               activation=tf.nn.tanh)
                    self.value_dense_2 = tf.layers.dense(inputs=self.value_dense_1, units=64,
                                                               activation=tf.nn.tanh)
                    self.value = tf.layers.dense(inputs=self.value_dense_2,units=1,kernel_initializer=normalized_columns_initializer(1.0))

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
                        self.policy_loss_arg[key] = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_arg[key]+0.05, 1e-20, 1.05)) * self.advantages)

                    self.policy_loss = 0
                    for key, value in dict_actionsepc.items():
                        self.policy_loss += self.policy_loss_arg[key]

                    # Get gradients from local network using local losses
                    self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.var_norms = tf.global_norm(self.local_vars)

                    if 'UpdateGradsMethod' not in dic_hparas.keys() or dic_hparas['UpdateGradsMethod'] not in ['Integration','Divide']:
                        print('dic_hparas key error : UpdateGradsMethod')
                    if dic_hparas['UpdateGradsMethod'] == 'Integration':
                        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
                        # self.gradients - gradients of loss wrt local_vars
                        self.gradients = tf.gradients(self.loss,self.local_vars)

                        self.grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,0.5) # old=40 2018年09月10日修改，梯度太大

                        # Apply local gradients to global network
                        self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                        self.apply_grads = trainer.apply_gradients(zip(self.grads,self.global_vars))
                    elif dic_hparas['UpdateGradsMethod'] == 'Divide':
                        # policy
                        self.loss_policy = self.policy_loss - self.entropy * 0.01
                        self.local_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope+'/public') + tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,scope+'/policy')
                        self.gradients_policy = tf.gradients(self.loss_policy,self.local_policy_vars)
                        self.grads_policy,self.grad_norms_policy = tf.clip_by_global_norm(self.gradients_policy,0.5)
                        self.global_vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global'+'/public') + tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,'global'+'/policy')
                        self.apply_grads_policy = trainer.apply_gradients(zip(self.grads_policy,self.global_vars_policy))
                        #value
                        self.loss_value = self.value_loss
                        self.local_value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope+'/value')
                        self.gradients_value = tf.gradients(self.loss_value,self.local_value_vars)
                        self.grads_value,self.grad_norms_value = tf.clip_by_global_norm(self.gradients_value,0.5)
                        self.global_vars_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global'+'/value')
                        self.apply_grads_value = trainer_critic.apply_gradients(zip(self.grads_value,self.global_vars_value))
        except Exception as e:
            common.echosentence_color('error in AC_NetWork -> init: {}'.format(str(e)) )
            raise


## WORKER AGENT
class Worker():
    def __init__(self, name, trainer, trainer_critic,  model_path,log_path,roomrd_path, global_episodes,flag_rlaicolor,dic_mainparas,
                 flag_testai = False,testai_freq = 500,testai_runnums = 20,step_length = 100,history_policy_len = 5):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.roomrd_path = roomrd_path
        self.flag_rlaicolor = flag_rlaicolor
        self.trainer = trainer
        self.trainer_critic = trainer_critic
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
        self.testai_freq = testai_freq # 测试ai的频率
        self.testai_runnums = testai_runnums # 测试50场的平均胜率
        self.history_policy_len = history_policy_len #指训练价值网时所使用的历史策略网长度
        self.step_length = step_length
        self.episode_test_winrates = [] #测试的平均胜率
        self.episode_test_winscores = [] #测试的平均净胜分
        self.prob_dec_fn = lambda x : 1.0/(0.0009*x+1.1)
        print('Initializing environment #{}...'.format(self.number))
        self.env = wgenv.ENV(dic_mainparas = dic_mainparas,flag_rlaicolor=flag_rlaicolor)
        filterMetadata(self.env,dic_hparas['BattleMode'])
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
        self.local_AC = AC_Network(self.name, trainer, trainer_critic, self.nonspatial_size,self.map_size,dict_actionsepc)
        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, list_rollout, sess, gamma, bootstrap_value,updateParas = None):
        '''modify: 2018年11月09日 宋国瑞 添加updateParas 取值:['Policy','Value','None','Both'] None或Both表示两者都更新'''
        try:
            common.echosentence_color('--' * 20, 'red')
            common.echosentence_color('Start Train {}'.format(updateParas))

            for index in range(len(list_rollout)):
                rollout = numpy.array(list_rollout[index])

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
                if index == 0:
                    feed_dict = {self.local_AC.target_v: discounted_rewards,
                                 self.local_AC.inputs_spatial_screen: numpy.stack(s_np_mapFeature).reshape(-1, self.map_size[0],self.map_size[1],self.map_size[2]),
                                 self.local_AC.inputs_nonspatial: numpy.stack(s_np_bopFeature).reshape(-1, self.nonspatial_size),
                                 self.local_AC.advantages: advantages}

                    for key,value in actions_arg_stack.items():
                        feed_dict[self.local_AC.actions_arg[key]] = value
                else:
                    feed_dict[self.local_AC.target_v] = numpy.concatenate((feed_dict[self.local_AC.target_v],discounted_rewards),axis = 0)
                    np_mapFeature = numpy.stack(s_np_mapFeature).reshape(-1, self.map_size[0],self.map_size[1],self.map_size[2])
                    np_bopFeature = numpy.stack(s_np_bopFeature).reshape(-1, self.nonspatial_size)
                    feed_dict[self.local_AC.inputs_spatial_screen] = numpy.concatenate((feed_dict[self.local_AC.inputs_spatial_screen], np_mapFeature), axis=0)
                    feed_dict[self.local_AC.inputs_nonspatial] = numpy.concatenate((feed_dict[self.local_AC.inputs_nonspatial], np_bopFeature), axis=0)
                    feed_dict[self.local_AC.advantages] = numpy.concatenate((feed_dict[self.local_AC.advantages], advantages), axis=0)
                    for key,value in actions_arg_stack.items():
                        feed_dict[self.local_AC.actions_arg[key]] += value
            if dic_hparas['UpdateGradsMethod'] == 'Integration':
                v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                   self.local_AC.policy_loss,
                                                   self.local_AC.entropy,
                                                   self.local_AC.grad_norms,
                                                   self.local_AC.var_norms,
                                                   self.local_AC.apply_grads],
                                                  feed_dict=feed_dict)
            elif dic_hparas['UpdateGradsMethod'] == 'Divide':
                if updateParas == 'Policy':
                    v_l, p_l, e_l, g_n, v_n, _, = sess.run([self.local_AC.value_loss,
                                                       self.local_AC.policy_loss,
                                                       self.local_AC.entropy,
                                                       self.local_AC.grad_norms_policy,
                                                       self.local_AC.var_norms,
                                                       self.local_AC.apply_grads_policy,
                                                       ],feed_dict=feed_dict)
                elif updateParas == 'Value':
                    v_l, p_l, e_l, g_n, v_n, _, = sess.run([self.local_AC.value_loss,
                                                       self.local_AC.policy_loss,
                                                       self.local_AC.entropy,
                                                       self.local_AC.grad_norms_policy,
                                                       self.local_AC.var_norms,
                                                       self.local_AC.apply_grads_value,
                                                       ],feed_dict=feed_dict)
                else:
                    v_l, p_l, e_l, g_n, v_n, _, _, = sess.run([self.local_AC.value_loss,
                                                            self.local_AC.policy_loss,
                                                            self.local_AC.entropy,
                                                            self.local_AC.grad_norms_policy,
                                                            self.local_AC.var_norms,
                                                            self.local_AC.apply_grads_policy,
                                                            self.local_AC.apply_grads_value,
                                                            ], feed_dict=feed_dict)

            common.echosentence_color('--' * 20, 'red')
            return v_l / rollout.shape[0] / rollout.shape[1], p_l / rollout.shape[0] / rollout.shape[1], \
                   e_l / rollout.shape[0] / rollout.shape[1], g_n, v_n
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
                list3_episode_buffer = []
                while True:   #not coord.should_stop():
                    # Download copy of parameters from global network
                    # prob = self.prob_dec_fn(episode_count) #用规则ai的概率
                    prob = 0
                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_values = []
                    # episode_frames = []
                    episode_reward = 0
                    episode_step_count = 0
                    episode_end = False

                    # 测试ai性能
                    if self.flag_testai and episode_count % self.testai_freq == 0 and episode_count >= 0:
                        winrate,winscore = self.test_ai(sess,self.testai_runnums)
                        self.episode_test_winrates.append(winrate)
                        self.episode_test_winscores.append(winscore)
                        common.echosentence_color('====================','blue')
                        common.echosentence_color('测试{}场的胜率为：{}% ；净胜分为 {}'.format(self.testai_runnums,winrate*100,winscore))
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
                            flag_thisvalid,flag_goon,reward,list2_chooseArg,list_npMapFea,list_npBopFea = self.env.stepFollowBaseAI()  # 依概率选择按照规则AI走,不参与训练
                            if flag_thisvalid: #生成有效动作
                                flag_baseai = True
                            else: #否则按照rl-ai选择动作
                                flag_baseai = False
                                policy_args, v = sess.run([self.local_AC.policy_arg, self.local_AC.value],
                                                          feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                                                                     self.local_AC.inputs_nonspatial: np_bopFeature})
                                l_nppolicy = []
                                for policy_key in self.l_action_keys:
                                    l_nppolicy.append(policy_args[policy_key][0, :])

                                flag_goon, reward, l_chooseArg = self.env.step(l_nppolicy[0], l_nppolicy[1],l_nppolicy[2])
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
                                reward += 20 # [0,200]

                        s1_np_mapFeature, s1_np_bopFeature = np_mapFeature, np_bopFeature
                        # if not episode_end:
                        #     s1_np_mapFeature, s1_np_bopFeature = np_mapFeature, np_bopFeature
                        # else:
                        #     s1_np_mapFeature, s1_np_bopFeature = s_np_mapFeature,s_np_bopFeature

                        # Append latest state to buffer
                        if not flag_baseai:
                            if l_chooseArg != None:
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
                        if len(episode_buffer) == self.step_length and not episode_end and episode_step_count != max_episode_length - 1:
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
                        if len(list3_episode_buffer) == self.history_policy_len:  # 保证buffer中最多有history_policy_len个策略生成的数据
                            list3_episode_buffer = list3_episode_buffer[1:]
                        list3_episode_buffer.append(episode_buffer)
                        # v1 = sess.run(self.local_AC.value,feed_dict={self.local_AC.inputs_spatial_screen: np_mapFeature,
                        #                          self.local_AC.inputs_nonspatial: np_bopFeature})
                        if dic_hparas['UpdateGradsMethod'] == 'Integration':
                            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0,
                                                                 updateParas='Both')  # 用v1替换0; 2018年09月11日重新替换为0
                        else:
                            v_l, p_l, e_l, g_n, v_n = self.train([episode_buffer], sess, gamma, 0,updateParas = 'Policy') # 用v1替换0; 2018年09月11日重新替换为0
                            v_l, p_l, e_l, g_n, v_n = self.train(list3_episode_buffer, sess, gamma, 0,
                                                             updateParas='Value')  # 用v1替换0; 2018年09月11日重新替换为0
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
                wincolor = 'RED' if red_score > 0 else 'BLUE'
                if self.env.obj_main.obj_pk.dic_paras['flag_qd_rm'] and random.random() < 0.1:
                    str_savefile = self.roomrd_path + 'BAK_Trainnums-{}_Testnums-{}-WinColor={}.xls'.format(int(_episodes[self.number])
                                                                            ,gamenum,wincolor)
                    wgsdata.record(self.env.obj_main,str_savefile)
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
        num_xd, num_plays, num_objcutility = 0, 1, 1
        strategy_id_r, strategy_id_b = 0, 1
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
                         'strategy_ids': (strategy_id_r,strategy_id_b),
                         'flag_action_cache': False,
                         'flag_qd_rm': True,
                         'flag_cache': False,
                         'flag_gpu': False,
                         'flag_afm': True,  # 两个AI都为BASE时，flag_afm=False; 否则为True
                         'flag_dllnum': 0,
                         'cuda_id': 0,
                         'flag_savestate': False,
                         'dic2_aiparas': {
                             'flag_color4acai': 0,
                             'red': {'type_ai': AI_QD.AI_QD_DEMO,
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

dic_hparas = {
    'BattleMode' : '6v6',
    'NetType': 'ResNet', #['AtariNet','ResNet']
    'UpdateGradsMethod':'Divide' #['Integration','Divide']
}
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
        exp_version = 'v10'
        num_workers = 1 # 多线程数 multiprocessing.cpu_count()
        flag_testai = True
        testai_freq = 500  # 测试ai的频率
        testai_runnums = 50  # 测试50场的平均胜率
        step_length = 200 # 推演多少步进行一次训练
        max_episode_length = 300
        gamma = 0.99  # Discount rate for advantage estimation and reward discounting
        history_policy_len = 5 # 指训练价值网时所使用的历史策略网长度
        str_ai = 'red' if flag_rlaicolor == 0 else 'blue'
        save_path = '../../../a3c-explog/'+ dic_hparas['BattleMode']+'_'+str_ai+'_'+exp_version+'/'
        model_path = save_path + 'model/'
        log_path = save_path + 'train_0/'
        roomrd_path = save_path + 'roomrecord/'
        dic_mainparas = defmainparas()

        tf.reset_default_graph()
        if not os.path.exists(model_path):
            load_model = False
            os.makedirs(model_path)
            os.makedirs(roomrd_path)

        common.echosentence_color('BattleMode = ' + dic_hparas['BattleMode'] ,'yellow')
        print('Initializing temporary environment to retrive action_spec...')
        env = wgenv.ENV(dic_mainparas = dic_mainparas,flag_rlaicolor=flag_rlaicolor)
        filterMetadata(env,dic_hparas['BattleMode'])
        wgfeature.cfunRegister(env.obj_main)

        wgfeature.calOneTimeFeature(env.obj_main)
        wgfeature.calUpdateFeature(env.obj_main)

        nonspatial_size = env.obj_main.dic_metadata['np_bopFeature'].shape[0]
        tmp_map_size = env.obj_main.dic_metadata['np_mapFeature'].shape

        map_size = (tmp_map_size[1], tmp_map_size[2], tmp_map_size[0])
        # if dic_hparas['NetType'] == 'ResNet':
        #     map_size = (tmp_map_size[1], tmp_map_size[1], tmp_map_size[0])
        l_action_keys = ['who', 'what', 'how']
        dict_actionsepc = {l_action_keys[0]: env.getWhoNums(), l_action_keys[1]: env.getWhatNums(),l_action_keys[2]: env.getHowNums()}
        print('Initializing temporary environment to retrive action_spec done...')

        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            trainer_critic = tf.train.AdamOptimizer(learning_rate=3e-4)
            master_network = AC_Network('global', None, None, nonspatial_size,map_size,dict_actionsepc)  # Generate global network
            global _max_score, _running_avg_score, _steps, _episodes
            _max_score = -200
            _running_avg_score = 0
            _steps = numpy.zeros(num_workers)
            _episodes = numpy.zeros(num_workers)
            workers = []
            # Create worker classes
            for i in range(num_workers):
                tmp_dic_mainparas = copy.deepcopy(dic_mainparas)
                dic_mainparas['flag_dllnum'] = i
                workers.append(Worker(i, trainer, trainer_critic, model_path, log_path, roomrd_path,global_episodes, flag_rlaicolor,dic_mainparas,
                                      flag_testai=flag_testai,testai_freq=testai_freq,testai_runnums= testai_runnums,step_length=step_length
                                      ,history_policy_len = history_policy_len))
            saver = tf.train.Saver(max_to_keep=5)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            coord = tf.train.Coordinator()
            if load_model == True:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            # This is where the asynchronous magic happens
            # Start the "work" process for each worker in a separate thread
            # while(True):
            #     for worker in workers:
            #         worker.work(max_episode_length, gamma, sess, None, saver)
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                time.sleep(0.001)
                worker_threads.append(t)
            coord.join(worker_threads)
    except Exception as e:
        common.echosentence_color('error in test_rl.py-> main() : {}'.format(str(e)))
        env.__del__()
        raise
    except KeyboardInterrupt as k:
        common.echosentence_color('error in test_rl.py-> main() : {}'.format(str(k)))
        env.__del__()
        raise
#
