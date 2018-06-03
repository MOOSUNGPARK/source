import tensorflow as tf
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim
import Portpolio.Cifar10_classify.Model_HENet.config as cfg

class HENet():
    def __init__(self, name, label_cnt):
        self.name = name
        self.label_cnt = label_cnt
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope(self.name):
            with tf.name_scope('initialize_scope'):
                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='X_data')
                self.Y = tf.placeholder(dtype=tf.int64, shape=[None], name='Y_data')
                self.training = tf.placeholder(dtype=tf.bool, name='training')
                self.initial_learning_rate = tf.get_variable('initial_learning_rate', initializer=cfg.LEARNING_RATE, trainable=False)
                self.l2_reg_rate = tf.get_variable('l2_reg_rate', initializer=cfg.L2_REG_RATE, trainable=False)
                self.global_step = tf.train.get_or_create_global_step() #############################
                self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                                self.global_step,
                                                                cfg.DECAY_STEPS,
                                                                cfg.DECAY_RATE,
                                                                cfg.STAIRCASE, name='learning_rate')

            def convlayer(name, l, num_filter, stride) :

                with tf.variable_scope(name):

                    l = slim.conv2d(l,
                                    num_outputs = num_filter,
                                    kernel_size = 3,
                                    stride = stride,
                                    scope = name + '_conv'
                                    )
                    l = slim.batch_norm(l, scope = name + '_batchnorm')

                    return l


            def channel_shuffle(name, l, num_group):

                with tf.variable_scope(name):

                    _, h, w, c = l.get_shape().as_list()
                    _l = tf.reshape(l, [-1, h, w, num_group, c // num_group])
                    _l = tf.transpose(_l, [0, 1, 2, 4, 3])
                    l = tf.reshape(_l, [-1, h, w, c])

                    return l

            def group_conv(name, l, num_filter, num_group, kernel_size, stride, activation_fn = True, padding = 'SAME'):

                with tf.variable_scope(name):

                    in_channel_per_group = l.get_shape().as_list()[3] // num_group
                    out_channel_per_group = num_filter // num_group
                    grouped_channel_list = []

                    for i in range(num_group):
                        _l = slim.conv2d(l[:, :, :, i*in_channel_per_group : i*in_channel_per_group + in_channel_per_group],
                                         num_outputs = out_channel_per_group,
                                         kernel_size = kernel_size,
                                         stride = stride,
                                         padding = padding,
                                         scope = name + str(i))
                        grouped_channel_list.append(_l)

                    l = tf.concat(grouped_channel_list, axis=-1, name='concat_channel')

                    if activation_fn :
                        l = slim.batch_norm(l, scope = name + '_batchnorm')
                    else :
                        l = slim.batch_norm(l, activation_fn = None, scope = name + '_batchnorm')

                    return l

            def h_layer(name, l, num_filter, group_m, group_n) :

                with tf.variable_scope(name):

                    l = group_conv(name = name + '_1st_Gconv',
                                   l = l,
                                   num_filter = num_filter,
                                   num_group = group_m,
                                   kernel_size = 1,
                                   stride = 1,
                                   activation_fn = False)

                    l = channel_shuffle(name, l, group_m)

                    l = group_conv(name = name + '_2nd_Gconv',
                                   l =  l,
                                   num_filter = num_filter,
                                   num_group = group_n,
                                   kernel_size = 3,
                                   stride = 1,
                                   activation_fn = True)
                    return l

            def s1_block(name, l, num_filter, group_m, group_n, repeat):

                with tf.variable_scope(name) :

                    X = tf.identity(l)

                    for i in range(repeat):
                        HL = h_layer(name = name + '_hlayer{}'.format(i+1),
                                     l = l,
                                     num_filter = num_filter,
                                     group_m = group_m,
                                     group_n = group_n)

                        X = X + HL
                        l = tf.concat([HL, X], axis = 3)

                    return l


            def s2_block(name, l, num_filter, group_m, group_n) :

                with tf.variable_scope(name) :

                    l = group_conv(name = name + '_1st_Gconv',
                                   l = l,
                                   num_filter = num_filter // 2,
                                   num_group = group_m,
                                   kernel_size = 3,
                                   stride = 2,
                                   activation_fn = False)

                    l = channel_shuffle(name, l = l, num_group = group_m)

                    l = group_conv(name = name + '_2nd_Gconv',
                                   l = l,
                                   num_filter = num_filter,
                                   num_group = group_n,
                                   kernel_size = 1,
                                   stride = 1,
                                   activation_fn = True)

                    return l

            def stage(name, l, filter_in, filter_out, group_m, group_n, repeat, last_stage = False) :

                with tf.variable_scope(name) :

                    if last_stage :

                        _, h, _, _ = l.get_shape().as_list()

                        l = group_conv(name = name + 'last_stage1',
                                       l = l,
                                       num_filter = filter_in,
                                       num_group = group_m,
                                       kernel_size = h,
                                       stride = 1,
                                       activation_fn = False,
                                       padding = 'VALID')

                        l = channel_shuffle(name, l = l, num_group = group_m)

                        l = group_conv(name = name + 'last_stage2',
                                       l = l,
                                       num_filter = filter_out,
                                       num_group = group_n,
                                       kernel_size = 1,
                                       stride = 1,
                                       activation_fn = True)

                    else :

                        l = s1_block(name = name + 's1_block',
                                     l = l,
                                     num_filter = filter_in,
                                     group_m = group_m,
                                     group_n = group_n,
                                     repeat = repeat)

                        l = s2_block(name = name + 's2_block',
                                     l = l,
                                     num_filter = filter_out,
                                     group_m = group_m,
                                     group_n = group_n)

                    return l



            def fclayer(name, l, num_filter, out_layer=False, dropout_yn=False):

                with tf.variable_scope(name):
                    l = slim.fully_connected(l, num_filter, activation_fn=None, scope = name + '_fc')

                    if not out_layer:
                        l = slim.batch_norm(inputs=l, scope = name + '_batchnorm')

                        if dropout_yn:
                            l = slim.dropout(l, scope=name + '_dropout')
                return l


            def cnn_model():

                with tf.variable_scope('mobilenet'):
                    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d, slim.fully_connected],
                                        activation_fn = None,
                                        weights_initializer  = variance_scaling_initializer(),
                                        biases_initializer = tf.zeros_initializer(),
                                        weights_regularizer = slim.l2_regularizer(self.l2_reg_rate)):
                        with slim.arg_scope([slim.batch_norm],
                                            is_training = self.training,
                                            scale = True,
                                            decay = cfg.BATCHNORM_DECAY_RATE,
                                            zero_debias_moving_mean = True,
                                            activation_fn = self.select_activation_fn(cfg.ACTIVATION_FN),
                                            fused = True):

                            l = convlayer('conv0', self.X, 24, 1)
                            print(l)
                            l = stage('stage1', l, 24, 48, 6, 4, 3)
                            print(l)
                            l = stage('stage2', l, 48, 96, 8, 6, 3)
                            print(l)
                            l = stage('stage3', l, 96, 96, 12, 8, 3)
                            print(l)
                            l = stage('stage4', l, 96, 192, 12, 8, 1, last_stage = True)
                            print(l)
                            l = tf.reshape(l, shape=[-1, 192])
                            print(l)
                            logits = fclayer('fc5', l, self.label_cnt, out_layer=True)

                return logits

            self.logits = cnn_model()
            self.softmax_logits = tf.nn.softmax(self.logits)
            slim.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.Y, scope='softmax_crossentropy')

        self.total_loss = tf.losses.get_total_loss()
        self.optimizer = self.select_optimizer(cfg.OPTIMIZER)
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y),
                                               dtype=tf.float32), name='accuracy')

    def swish(self, x):
        return x * tf.nn.sigmoid(x, 'swish')

    def select_optimizer(self, opt):
        if opt.lower() == 'adam' :
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        elif opt.lower() == 'rmsprop' :
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

    def select_activation_fn(self, fn):
        if fn.lower() == 'swish' :
            return self.swish

        elif fn.lower() == 'elu' :
            return tf.nn.elu