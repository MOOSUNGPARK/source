import tensorflow as tf
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim
import Portpolio.Cifar10_classify.Model_mobilenet.config as cfg

class Mobilenet():
    def __init__(self, name, label_cnt):
        self.name = name
        self.label_cnt = label_cnt
        self.width_multiplier = cfg.WIDTH_MULTIPLIER
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

            def depthwise_convlayer(name, l, num_filter, width_multiplier, stride, dropout_yn=False):

                with tf.variable_scope(name):
                    multiplied_filter = int(num_filter * width_multiplier)

                    # Depthwise
                    l = slim.separable_convolution2d(l,
                                                     num_outputs = None,
                                                     kernel_size = 3,
                                                     depth_multiplier = 1,
                                                     stride = stride,
                                                     scope = name + '_depthwise'
                                                     )
                    l = slim.batch_norm(l, scope = name + '_batchnorm1')

                    # Pointwise
                    l = slim.conv2d(l,
                                    num_outputs = multiplied_filter,
                                    kernel_size = 1,
                                    scope = name + '_pointwise'
                                    )
                    l = slim.batch_norm(l, scope = name + '_batchnorm2')

                    if dropout_yn:
                        l = slim.dropout(l, scope=name + '_dropout')

                return l

            def convlayer(name, l, num_filter, width_multiplier, stride, dropout_yn=False) :

                with tf.variable_scope(name):
                    multiplied_filter = int(num_filter * width_multiplier)

                    l = slim.conv2d(l,
                                    num_outputs = multiplied_filter,
                                    kernel_size = 3,
                                    stride = stride,
                                    scope = name + '_conv'
                                    )
                    l = slim.batch_norm(l, scope = name + '_batchnorm')

                    if dropout_yn :
                        l = slim.dropout(l, scope = name + '_dropout')

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
                                        weights_initializer = variance_scaling_initializer(),
                                        biases_initializer = tf.zeros_initializer(),
                                        weights_regularizer = slim.l2_regularizer(self.l2_reg_rate)):
                        with slim.arg_scope([slim.batch_norm],
                                            is_training = self.training,
                                            scale = True,
                                            decay = cfg.BATCHNORM_DECAY_RATE,
                                            zero_debias_moving_mean = True,
                                            activation_fn = self.select_activation_fn(cfg.ACTIVATION_FN),
                                            fused = True):
                            with slim.arg_scope([slim.dropout],
                                                is_training = self.training,
                                                keep_prob = cfg.DROPOUT_KEEP_PROB):

                                l = convlayer('conv0', self.X, 32, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw1', l, 64, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw2', l, 128, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw3', l, 128, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw4', l, 256, self.width_multiplier, 2, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw5', l, 256, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw6', l, 512, self.width_multiplier, 2, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw7', l, 512, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw8', l, 512, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw9', l, 512, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw10', l, 512, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw11', l, 512, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw12', l, 512, self.width_multiplier, 1, dropout_yn=True)
                                l = depthwise_convlayer('conv_dw13', l, 1024, self.width_multiplier, 2, dropout_yn=True)
                                l = avg_pool2d(l, kernel_size=4, stride=1, padding='VALID', scope='avg_pool14')
                                l = tf.reshape(l, shape=[-1, int(1 * 1 * 1024 * self.width_multiplier)])
                                logits = fclayer('fc15', l, self.label_cnt, out_layer=True)

                return logits

            self.logits = cnn_model()
            self.softmax_logits = tf.nn.softmax(self.logits)
            slim.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.Y, scope='softmax_crossentropy')

        self.total_loss = tf.losses.get_total_loss()
        self.optimizer = self.select_optimizer(cfg.OPTIMIZER)
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y), dtype=tf.float32),
                                       name='accuracy')

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


