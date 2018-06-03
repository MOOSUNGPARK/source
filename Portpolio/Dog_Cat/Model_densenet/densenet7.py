import tensorflow as tf
from tensorflow.contrib.layers import *


# elu / Adam / lr = 0.004 / l2_reg = 1e-4 / dropout_rate = 0.2

class Model():
    def __init__(self, sess, depth, label_cnt):
        self.Sess = sess
        self.N = int((depth - 4) / 3)
        self.Growth_Rate = 12
        self.Compression_Factor = 0.5
        self.Label_Cnt = label_cnt
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('initialize_scope'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 126 * 126], name='X_data')
            X_img = tf.reshape(self.X, shape=[-1, 126, 126, 1])
            self.Y = tf.placeholder(dtype=tf.int64, shape=[None, self.Label_Cnt], name='Y_data')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
            self.learning_rate = tf.get_variable('learning_rate', initializer=0.004, trainable=False)
            self.l2_reg_rate = tf.get_variable('l2_reg', initializer=1e-4, trainable=False)

        def conv(l, kernel, channel, stride, padding='SAME'):
            return conv2d(inputs=l, num_outputs=channel, kernel_size=kernel, stride=stride, padding=padding,
                          weights_initializer=variance_scaling_initializer(), biases_initializer=None,
                          weights_regularizer=l2_regularizer(self.l2_reg_rate), activation_fn=tf.nn.elu)

        def add_convlayer(name, l, kernel, channel, stride=1, padding='SAME', pool_yn=True, dropout_yn=True):
            with tf.variable_scope(name):
                l = conv(l, kernel, channel, stride, padding)
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                if pool_yn:
                    l = max_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
                if dropout_yn:
                    l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            return l

        def add_layer(name, l):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = conv(l, 1, 4 * self.Growth_Rate, 1)
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)

                '''basic dense layer'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = conv(l, 3, self.Growth_Rate, 1)
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = conv(l, 3, int(in_channel * self.Compression_Factor), 1)
                l = avg_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            return l

        def add_fclayer(name, l, num_input, num_output, out_layer=False):
            with tf.variable_scope(name):
                w = tf.get_variable(name=name + '_w', shape=[num_input, num_output], dtype=tf.float32,
                                    initializer=variance_scaling_initializer(),
                                    regularizer=l2_regularizer(self.l2_reg_rate))
                b = tf.Variable(tf.constant(value=0.001, shape=[num_output], name=name + '_b'))
                l = tf.matmul(l, w) + b
                if not out_layer:
                    l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True,
                                   is_training=self.training)
                    l = self.parametric_relu(l, name + '_prelu')
            return l

        def dense_net():
            l = add_convlayer('Conv1_sub1', X_img, 3, 16, padding='VALID', pool_yn=False, dropout_yn=False)
            l = add_convlayer('Conv1_sub2', l, 3, 16, padding='VALID', pool_yn=False, dropout_yn=False)
            l = add_convlayer('Conv1_sub3', l, 3, 16, padding='VALID', pool_yn=False, dropout_yn=False)
            l = max_pool2d(inputs=l, kernel_size=[4, 4], stride=4, padding='SAME')
            print('1',l)
            with tf.variable_scope('dense_block1'):
                pl = tf.identity(l)
                for idx in range(self.N):
                    l = add_layer('dense_layer1_{}'.format(idx), pl)
                    pl = tf.concat([pl, l], axis=3)
                l = add_transition('transition1', pl)
            print('2',l)
            with tf.variable_scope('dense_block2'):
                pl = tf.identity(l)
                for idx in range(self.N):
                    l = add_layer('dense_layer2_{}'.format(idx), l)
                    pl = tf.concat([pl, l], axis=3)
                l = add_transition('transition2', pl)
            print('3',l)
            with tf.variable_scope('dense_block3'):
                pl = tf.identity(l)
                for idx in range(self.N):
                    l = add_layer('dense_layer3_{}'.format(idx), l)
                    pl = tf.concat([pl, l], axis=3)
            print('4',pl)
            l = avg_pool2d(inputs=pl, kernel_size=[8, 8], stride=1, padding='VALID')  # kernel_size 변경
            print('5',l)
            l = tf.reshape(l, shape=[-1, 1 * 1 * 256])
            logits = add_fclayer('out_layer', l, 1 * 1 * 256, self.Label_Cnt, out_layer=True)

            return logits

        self.logits = dense_net()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([loss] + reg_losses, name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.Sess.run(self.logits, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x_test, y_test):
        return self.Sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.training: False, self.dropout_rate: 1.0})

    def train(self, x_data, y_data):
        return self.Sess.run([self.accuracy, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data,
                                                                         self.training: True, self.dropout_rate: 0.2})

    def validation(self, x_test, y_test):
        return self.Sess.run([self.loss, self.accuracy],
                             feed_dict={self.X: x_test, self.Y: y_test, self.training: False, self.dropout_rate: 1.0})

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
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError(
                'number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units  # m
        shape += [num_channels // num_units]  # k
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs


