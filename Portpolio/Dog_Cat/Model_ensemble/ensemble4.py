import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import *

# prelu / Adam / lr = 0.001 / l2_reg_rate = 1e-4

class Model():
    def __init__(self, sess, name, label_cnt, lr, dr):
        self.Sess = sess
        self.name = name
        self.lr = lr
        self.dr = dr
        self.Label_Cnt = label_cnt
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('initialize_scope'):
                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 126 * 126], name='X_data')
                X_img = tf.reshape(self.X, shape=[-1, 126, 126, 1])
                self.Y = tf.placeholder(dtype=tf.int64, shape=[None, self.Label_Cnt], name='Y_data')
                self.training = tf.placeholder(dtype=tf.bool, name='training')
                self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
                self.learning_rate = tf.get_variable('learning_rate', initializer=self.lr, trainable=False)
                self.l2_reg_rate = tf.get_variable('l2_reg', initializer=1e-4, trainable=False)

            def conv(l, kernel, channel_out, stride, padding='SAME'):
                return conv2d(inputs=l, num_outputs=channel_out, kernel_size=kernel, stride=stride, padding=padding,
                              activation_fn=tf.nn.elu, weights_initializer=variance_scaling_initializer(),
                              biases_initializer=None, weights_regularizer=l2_regularizer(self.l2_reg_rate))

            def conv2(l, kernel, channel_in, channel_out, stride, padding='SAME'):
                w = tf.get_variable(name='W1_sub', shape=[kernel, kernel, channel_in, channel_out],
                                              dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                return tf.nn.conv2d(input=l, filter=w, strides=[stride, stride, stride, stride], padding=padding)

            def add_convlayer(name, l, kernel, channel, stride=1, padding='SAME', pool_yn=True, dropout_yn=True):
                with tf.variable_scope(name):
                    l = conv(l, kernel, channel, stride, padding)
                    l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                    if pool_yn:
                        l = max_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
                    if dropout_yn:
                        l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
                return l

            def add_convlayer2(name, l, kernel, channel_in, channel_out, stride=1, padding='SAME', pool_yn=True, dropout_yn=True):
                with tf.variable_scope(name):
                    l = conv2(l, kernel, channel_in, channel_out, stride, padding)
                    l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training, scope=name)
                    l = self.parametric_relu(l, name)

                    if pool_yn:
                        l = max_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
                    if dropout_yn:
                        l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
                return l

            def add_fclayer(name, l, num_input, num_output, out_layer=False):
                with tf.variable_scope(name):
                    w = tf.get_variable(name=name+'_w', shape=[num_input, num_output], dtype=tf.float32, initializer=variance_scaling_initializer(),
                                        regularizer=l2_regularizer(self.l2_reg_rate))
                    b = tf.Variable(tf.constant(value=0.001, shape=[num_output], name=name+'_b'))
                    l = tf.matmul(l, w) + b
                    if not out_layer:
                        l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                        l = self.parametric_relu(l, name+'_prelu')
                return l

            def cnn_model():
                l = add_convlayer2('Conv1_sub1', X_img, 3, 1, 20, padding='VALID', pool_yn=False, dropout_yn=False)
                l = add_convlayer2('Conv1_sub2', l, 3, 20, 20, padding='VALID', pool_yn=False, dropout_yn=False)
                l = add_convlayer2('Conv1_sub3', l, 3, 20, 20, padding='VALID') # 120 * 120 -> 60 * 60
                l = add_convlayer2('Conv2', l, 3, 20, 40) # 60 * 60 * 20 -> 30 * 30 * 40
                l = add_convlayer2('Conv3', l, 3, 40, 80)
                l = add_convlayer2('Conv4', l, 3, 80, 160)
                l = add_convlayer2('Conv5', l, 3, 160, 320) # 4 * 4 * 320 shape= [None, 4, 4, 320]
                l = tf.reshape(l, shape = [-1, 4 * 4 * 320])
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
                l = add_fclayer('Fc1', l, 4 * 4 * 320, 1000)
                l = add_fclayer('Fc2', l, 1000, 1000)
                logits = add_fclayer('Fc3', l, 1000, self.Label_Cnt, out_layer=True)
                return logits

            self.logits = cnn_model()
            self.softmax_logits = tf.nn.softmax(self.logits)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            loss = tf.reduce_mean(loss, name='cross_entropy_loss')
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name)

        self.loss = tf.add_n([loss] + reg_losses, name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits,1), tf.arg_max(self.Y,1)), dtype=tf.float32))

    def parametric_relu(self, x, name):
        alphas = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.01),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg

    def max_out(self, inputs, num_units, axis=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError(
                'number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units  # m
        shape += [num_channels // num_units]  # k
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

    def predict(self, x_test):
        return self.Sess.run(self.softmax_logits, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x_test, y_test):
        return self.Sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False, self.dropout_rate: 1.0})

    def train(self, x_data, y_data):
        return self.Sess.run([self.accuracy, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data,
                                                                                    self.training: True, self.dropout_rate: self.dr})

    def validation(self, x_test, y_test):
        return self.Sess.run([self.loss, self.accuracy], feed_dict={self.X: x_test, self.Y: y_test, self.training: False, self.dropout_rate: 1.0})

def ensemble_accuracy(predict, y_test):
    return np.mean(np.equal(np.argmax(predict, 1), np.argmax(y_test, 1)).astype(float))
