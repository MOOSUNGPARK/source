import tensorflow as tf
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim
import config as cfg

class densenet():
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


            # BN - Relu - Conv
            def convlayer(name, l, kernel_size, num_filter, stride) :

                with tf.variable_scope(name):
                    l = slim.batch_norm(l, scope = name + '_batchnorm')
                    l = slim.conv2d(l,
                                    num_outputs = num_filter,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    scope = name + '_conv'
                                    )
                    return l

            def dense_layer(name, l):

                with tf.variable_scope(name):
                    l = convlayer(name + '_bottle', l, 1, 4 * cfg.GROWTH_RATE, 1)
                    l = slim.dropout(l, scope = name + '_bottle_dropout1')

                    l = convlayer(name + '_notbottle', l, 3, cfg.GROWTH_RATE, 1)
                    l = slim.dropout(l, scope=name + '_bottle_dropout2')

                return l

            def transition_layer(name, l, num_filter):
                l = slim.batch_norm(l, activation_fn = None, scope = name + '_batchnorm')
                l = slim.conv2d(l,
                                num_outputs = cfg.THETA * num_filter,
                                kernel_size = 1,
                                stride = 1,
                                scope = name + '_conv')

                l = tf.nn.avg_pool(l,
                                   ksize=2,
                                   strides=2,
                                   padding='SAME',
                                   name = name + '_Avgpool'
                                   )
                return l

            def dense_block(name, l, num_layer):
                hl = tf.identity(l)


                for idx in range(num_layer):
                    l = dense_layer(name + '_layer{}'.format(idx), hl)
                    hl = tf.concat([hl, l], axis=3)

                return hl


            def global_avgpooling(name, l):

                with tf.variable_scope(name):
                    ksize = l.get_shape().as_list()[1]
                    num_filter= l.get_shape().as_list()[-1]

                    # gap_filter shape = [h,w,input_filter, output_filter]
                    gap_filter = tf.get_variable(name='gap_filter',
                                                 shape=[1, 1, num_filter, cfg.LABEL_CNT],
                                                 dtype=tf.float32,
                                                 initializer=variance_scaling_initializer()
                                                 )
                    l = tf.nn.conv2d(l,
                                     filter =gap_filter,
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     name= name + '_GAP_Conv'
                                     )
                    l = tf.nn.avg_pool(l,
                                       ksize=[1, ksize, ksize, 1],
                                       strides=[1, 1, 1, 1],
                                       padding='VALID',
                                       name = name + '_GAP_Avgpool'
                                       )
                    l = tf.reduce_mean(l, axis=[1,2])

                return l

            def cnn_model():

                with tf.variable_scope('resnet'):
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
                            # def residual_block(name, l, num_filter, stride):


                            l = convlayer('conv0', self.X, 3, 24, 1)
                            print(l)

                            # def dense_block(name, l, num_layer):
                            # def transition_layer(name, l, num_filter):

                            l = dense_block('dense_block1', l, 6)
                            print(l)
                            l = transition_layer('transition1', l, 96)
                            print(l)

                            l = dense_block('dense_block2', l, 12)
                            print(l)
                            l = transition_layer('transition1', l, 196)
                            print(l)

                            l = dense_block('dense_block3', l, 24)
                            print(l)
                            l = transition_layer('transition1', l, 232)
                            print(l)

                            l = dense_block('dense_block4', l, 16)
                            print(l)
                            l = transition_layer('transition1', l, 260)
                            print(l)

                            logits = global_avgpooling('GAP', l)
                            print(logits)

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