import tensorflow as tf
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim
import Portpolio.Dog_Cat.Model_vggnet.renewal.config as cfg

class Vggnet():
    def __init__(self, name, label_cnt):
        self.name = name
        self.label_cnt = label_cnt
        self._build_graph()

    def _build_graph(self):
        def convlayer(name, l, num_filter, padding='SAME', pool_yn=False, dropout_yn=False):

            with tf.variable_scope(name):
                l = slim.conv2d(l,
                                num_outputs=num_filter,
                                padding=padding,
                                kernel_size=3,
                                scope=name + '_conv'
                                )
                l = slim.batch_norm(l, scope=name + '_batchnorm')

                if pool_yn:
                    l = slim.max_pool2d(l, kernel_size=2, padding='SAME', scope=name + '_maxpool')

                if dropout_yn:
                    l = slim.dropout(l, scope=name + '_dropout')

            return l

        def fclayer(name, l, num_filter, out_layer=False, dropout_yn=False):

            with tf.variable_scope(name):
                l = slim.fully_connected(l, num_filter, activation_fn=None, scope=name + '_fc')

                if not out_layer:
                    l = slim.batch_norm(inputs=l, scope=name + '_batchnorm')

                    if dropout_yn:
                        l = slim.dropout(l, scope=name + '_dropout')
            return l



        with tf.variable_scope(self.name):
            with tf.name_scope('initialize_scope'):
                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 126 * 126], name='X_data')
                X_img = tf.reshape(self.X, shape=[-1, 126, 126, 1])
                self.Y = tf.placeholder(dtype=tf.int64, shape=[None, self.label_cnt], name='Y_data')
                self.training = tf.placeholder(dtype=tf.bool, name='training')
                self.initial_learning_rate = tf.get_variable('learning_rate', initializer=cfg.LEARNING_RATE, trainable=False)
                self.l2_reg_rate = tf.get_variable('l2_reg_rate', initializer=cfg.L2_REG_RATE, trainable=False)
                self.global_step = tf.train.get_or_create_global_step() #############################
                self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                                self.global_step,
                                                                cfg.DECAY_STEPS,
                                                                cfg.DECAY_RATE,
                                                                cfg.STAIRCASE, name='learning_rate')





            def cnn_model():

                with tf.variable_scope('vggnet'):
                    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d, slim.fully_connected],
                                        activation_fn = None,
                                        weights_initializer  = xavier_initializer(),
                                        biases_initializer = tf.zeros_initializer(),
                                        weights_regularizer = slim.l2_regularizer(self.l2_reg_rate)):
                        with slim.arg_scope([slim.batch_norm],
                                            is_training = self.training,
                                            decay=cfg.BATCHNORM_DECAY_RATE,
                                            activation_fn = self.select_activation_fn(cfg.ACTIVATION_FN),
                                            fused = True):
                            with slim.arg_scope([slim.dropout],
                                                is_training = self.training,
                                                keep_prob = cfg.DROPOUT_KEEP_PROB):

                                l = convlayer('conv0_0', X_img, 20, padding='VALID')
                                l = convlayer('conv0_1', l, 20, padding='VALID')
                                l = convlayer('conv0_2', l, 20, padding='VALID', pool_yn=True, dropout_yn=True)

                                l = convlayer('conv1_0', l, 40)
                                l = convlayer('conv1_1', l, 40, pool_yn=True, dropout_yn=True)

                                l = convlayer('conv2_0', l, 80)
                                l = convlayer('conv2_1', l, 80, pool_yn=True, dropout_yn=True)

                                l = convlayer('conv3_0', l, 160)
                                l = convlayer('conv3_1', l, 160)
                                l = convlayer('conv3_2', l, 160, pool_yn=True, dropout_yn=True)

                                l = convlayer('conv4_0', l, 320)
                                l = convlayer('conv4_1', l, 320)
                                l = convlayer('conv4_2', l, 320, pool_yn=True, dropout_yn=True)

                                l = tf.reshape(l, shape=[-1, 4 * 4 * 320])
                                l = fclayer('fc14', l, 1000)
                                l = fclayer('fc15', l, 1000)
                                logits = fclayer('fc16', l, self.label_cnt, out_layer=True)

                return logits

            self.logits = cnn_model()
            self.softmax_logits = tf.nn.softmax(self.logits)
            slim.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.Y)

        self.total_loss = tf.losses.get_total_loss()
        self.optimizer = self.select_optimizer(cfg.OPTIMIZER)
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
                                                        tf.argmax(self.Y, 1)), dtype=tf.float32))



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

        elif fn.lower() == 'relu' :
            return tf.nn.relu