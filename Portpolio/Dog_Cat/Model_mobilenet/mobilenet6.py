import tensorflow as tf
from tensorflow.contrib.layers import *
import tensorflow.contrib.slim as slim


class Model():
    def __init__(self, sess, name, label_cnt):
        self.Sess = sess
        self.name = name
        self.Label_Cnt = label_cnt
        self.width_multiplier = 0.5
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('initialize_scope'):
                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 126 * 126], name='X_data')
                X_img = tf.reshape(self.X, shape=[-1, 126, 126, 1])
                self.Y = tf.placeholder(dtype=tf.int64, shape=[None, self.Label_Cnt], name='Y_data')
                self.training = tf.placeholder(dtype=tf.bool, name='training')
                self.learning_rate = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
                self.l2_reg_rate = tf.get_variable('l2_reg_rate', initializer=1e-4, trainable=False)

            def depthwise_convlayer(name, l, num_filter, width_multiplier, stride):
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

                return l

            def convlayer(name, l, num_filter, width_multiplier, stride) :
                with tf.variable_scope(name):
                    multiplied_filter = int(num_filter * width_multiplier)

                    l = slim.conv2d(l,
                                    num_outputs = multiplied_filter,
                                    kernel_size = 3,
                                    stride = stride,
                                    scope = name + '_conv'
                                    )
                    l = slim.batch_norm(l, scope = name + '_batchnorm')

                return l


            def fclayer(name, l, num_output, out_layer=False):
                with tf.variable_scope(name):
                    l = slim.fully_connected(l, num_output, activation_fn=None, scope='fc_16')
                    if not out_layer:
                        l = slim.batch_norm(inputs=l, scope = name + '_batchnorm')

                return l


            def cnn_model():
                with tf.variable_scope('mobilenet'):
                    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d, slim.fully_connected],
                                        activation_fn = None,
                                        weights_initializer  = xavier_initializer(),
                                        biases_initializer = tf.zeros_initializer(),
                                        weights_regularizer = slim.l2_regularizer(self.l2_reg_rate)):
                        with slim.arg_scope([slim.batch_norm],
                                            is_training = self.training,
                                            decay=0.9,
                                            activation_fn = self.swish,
                                            fused = True):
                            l = convlayer('conv0', X_img, 32, self.width_multiplier, 2)
                            l = depthwise_convlayer('conv_dw1', l, 64, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw2', l, 128, self.width_multiplier, 2)
                            l = depthwise_convlayer('conv_dw3', l, 128, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw4', l, 256, self.width_multiplier, 2)
                            l = depthwise_convlayer('conv_dw5', l, 256, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw6', l, 512, self.width_multiplier, 2)
                            l = depthwise_convlayer('conv_dw7', l, 512, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw8', l, 512, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw9', l, 512, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw10', l, 512, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw11', l, 512, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw12', l, 512, self.width_multiplier, 1)
                            l = depthwise_convlayer('conv_dw13', l, 1024, self.width_multiplier, 2)
                            l = avg_pool2d(l, kernel_size=4, stride=1, padding='VALID', scope='avg_pool14')
                            l = tf.reshape(l, shape=[-1, int(1 * 1 * 1024 * self.width_multiplier)])
                            logits = fclayer('fc15', l, self.Label_Cnt, out_layer=True)
                return logits

            self.logits = cnn_model()
            self.softmax_logits = tf.nn.softmax(self.logits)
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            slim.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.Y)
            # loss = tf.reduce_mean(loss, name='cross_entropy_loss')
            # slim.losses.add_loss(loss)
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name)

        self.total_loss = tf.losses.get_total_loss()

        # self.loss = tf.add_n([loss] + reg_losses, name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer)
        # self.final_loss = slim.learning.train(self.train_op,
        #                                  logdir='./',
        #                                  )


        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)), dtype=tf.float32))

    def swish(self, x):
        return x * tf.nn.sigmoid(x, 'swish')

    def predict(self, x_test):
        return self.Sess.run(self.softmax_logits,
                             feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.Sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.Sess.run([self.accuracy, self.train_op], feed_dict={self.X: x_data, self.Y: y_data,
                                                                         self.training: True})

    def validation(self, x_test, y_test):
        return self.Sess.run([self.train_op, self.accuracy],
                             feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

