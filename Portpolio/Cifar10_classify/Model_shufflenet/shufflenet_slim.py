import tensorflow as tf
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim
import Portpolio.Cifar10_classify.Model_shufflenet.config as cfg

class Shufflenet():
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
                    # l = slim.max_pool2d(l, kernel_size = 3, stride = stride, padding = 'SAME')

                    return l


            def channel_shuffle(name, l, num_group):

                with tf.variable_scope(name):

                    # _, h, w, c = l.shape
                    _, h, w, c = l.get_shape().as_list()
                    _l = tf.reshape(l, [-1, h, w, num_group, c // num_group])
                    _l = tf.transpose(_l, [0, 1, 2, 4, 3])
                    l = tf.reshape(_l, [-1, h, w, c])

                    return l

            def group_conv(name, l, num_filter, num_group, stride, activation_fn = True):

                with tf.variable_scope(name):

                    in_channel_per_group = l.get_shape().as_list()[3] // num_group
                    out_channel_per_group = num_filter // num_group
                    grouped_channel_list = []

                    for i in range(num_group):
                        _l = slim.conv2d(l[:, :, :, i*in_channel_per_group : i*in_channel_per_group + in_channel_per_group],
                                         num_outputs = out_channel_per_group,
                                         kernel_size = 1,
                                         stride = stride,
                                         scope = name + str(i))
                        grouped_channel_list.append(_l)

                    l = tf.concat(grouped_channel_list, axis=-1, name='concat_channel')

                    if activation_fn :
                        l = slim.batch_norm(l, scope = name + '_batchnorm')
                    else :
                        l = slim.batch_norm(l, activation_fn = None, scope = name + '_batchnorm')

                    return l

            def shufflenet_unit(name, l, num_filter, num_group, stride):

                with tf.variable_scope(name):

                    activation_fn = self.select_activation_fn(cfg.ACTIVATION_FN)

                    if stride != 1 : # repeat = 0 일 때
                        _, _, _, c = l.get_shape().as_list()
                        num_filter = num_filter - c # Residual 채널 수 c개 + Group 채널 수 = 최종 채널수(num_filter) 가 되어야 하므로

                    # Residual part
                    if stride != 1 : # repeat = 0 일 때
                        residual_layer = avg_pool2d(l, kernel_size=3, stride=2, padding='SAME', scope='residual_avgpool')

                    else : # repeat >= 1 일 때
                        residual_layer = l

                    # Group part
                    _group_layer = group_conv(name = name + '_1st_group_conv',
                                              l = l,
                                              num_filter = num_filter,
                                              num_group = num_group,
                                              stride = 1,
                                              activation_fn = True
                                              )

                    _shuffled_group_layer = channel_shuffle(name = name + '_channel_shuffle',
                                                           l = _group_layer,
                                                           num_group = num_group
                                                           )

                    _depthwise_conv_layer = slim.separable_convolution2d(_shuffled_group_layer,
                                                                        num_outputs = None,
                                                                        kernel_size = 3,
                                                                        depth_multiplier = 1,
                                                                        stride = stride,
                                                                        scope = name + '_depth_wise'
                                                                        )

                    _depthwise_conv_layer = slim.batch_norm(_depthwise_conv_layer,
                                                           activation_fn= None,
                                                           scope = name + '_batchnorm'
                                                           )

                    final_group_layer = group_conv(name = name + '_2nd_group_conv',
                                                   l = _depthwise_conv_layer,
                                                   num_filter = num_filter,
                                                   num_group = num_group,
                                                   stride = 1,
                                                   activation_fn = False
                                                   )

                    # Concat part
                    if stride != 1 : # repeat = 0 일 때
                        final_layer = activation_fn(tf.concat([residual_layer, final_group_layer], axis = 3))

                    else : # repeat >= 1 일 때
                        final_layer = activation_fn(residual_layer + final_group_layer)

                    return final_layer

            def stage(name, l, num_filter, num_group, repeat) :

                with tf.variable_scope(name) :

                    l = shufflenet_unit(name = name , l = l, num_filter = num_filter, num_group = num_group, stride = 2)

                    for i in range(repeat) :
                        l = shufflenet_unit(name = name + str(i), l = l, num_filter = num_filter, num_group = num_group, stride = 1)

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

                            l = convlayer('conv1', self.X, cfg.CONV1_CHANNEL, 2)
                            print('1',l)
                            l = stage('stage2', l, cfg.STAGE2_CHANNEL, cfg.GROUP, 3) # 12
                            print('2',l)
                            l = stage('stage3', l, cfg.STAGE3_CHANNEL, cfg.GROUP, 7) #
                            print('3',l)
                            l = stage('stage4', l, cfg.STAGE4_CHANNEL, cfg.GROUP, 3)
                            print('4',l)
                            l = avg_pool2d(l, kernel_size=2, stride=1, padding='VALID', scope='avg_pool9')
                            print('5',l)
                            l = tf.reshape(l, shape=[-1, cfg.STAGE4_CHANNEL])
                            print('6',l)
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