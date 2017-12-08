import tensorflow as tf
from tensorflow.contrib.layers import *

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
            self.learning_rate = tf.get_variable('learning_rate', initializer=0.01, trainable=False)

        def conv(l, kernel, channel, stride, padding='SAME'):
            return conv2d(inputs=l, num_outputs=channel, kernel_size=kernel, stride=stride, padding=padding,
                          weights_initializer=variance_scaling_initializer(), biases_initializer=None,
                          weights_regularizer=l2_regularizer(1e-4))

        def add_layer(name, l, block_number, idx):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = self.parametric_relu(l, 'Bottle' + str(block_number) + '_' + str(idx))
                l = conv(l, 1, 4 * self.Growth_Rate, 1)
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)

                '''basic dense layer'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = self.parametric_relu(l, 'Dense' +  str(block_number) + '_' + str(idx))
                l = conv(l, 3, self.Growth_Rate, 1)
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
                # l = tf.concat([c,l], axis=3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = self.parametric_relu(l, name + '_' + str(1))
                l = conv(l, 3, int(in_channel * self.Compression_Factor), 1)
                l = avg_pool2d(inputs=l, kernel_size=[2,2], stride=2, padding='SAME')
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            return l

        def dense_net():
            l = conv(X_img, 3, 16, 1)
            l = max_pool2d(inputs=l, kernel_size=[2,2], stride=2, padding='SAME')
            print(l)

            with tf.variable_scope('dense_block1'):
                pl = tf.identity(l)
                for idx in range(self.N):
                    l = add_layer('dense_layer1_{}'.format(idx), pl, 1, idx)
                    pl = tf.concat([pl,l], axis=3)
                l = add_transition('transition1', pl)

            with tf.variable_scope('dense_block2'):
                # pl = deepcopy(l)
                pl = tf.identity(l)
                for idx in range(self.N):
                    l = add_layer('dense_layer2_{}'.format(idx), l, 2, idx)
                    pl = tf.concat([pl,l], axis=3)
                l = add_transition('transition2', pl)

            with tf.variable_scope('dense_block3'):
                # pl = deepcopy(l)
                pl = tf.identity(l)
                for idx in range(self.N):
                    l = add_layer('dense_layer3_{}'.format(idx), l, 3, idx)
                    pl = tf.concat([pl,l], axis=3)
            print(pl)
            l = batch_norm(inputs=pl, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
            l = self.parametric_relu(l, 'output')
            l = max_pool2d(inputs=l, kernel_size=[2,2], stride=2, padding='SAME')
            l = avg_pool2d(inputs=l, kernel_size=[8,8], stride=1, padding='VALID') # kernel_size 변경
            l = tf.reshape(l, shape=[-1, 1 * 1 * 256])
            l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            # with tf.name_scope('fc_layer') as scope:
            #     W_fc = tf.get_variable(name='W_fc', shape=[1 * 1 * 256, self.Label_Cnt], dtype=tf.float32,
            #                                  initializer=variance_scaling_initializer())
            #     b_fc = tf.Variable(tf.constant(value=0.001, shape=[self.Label_Cnt], name='b_fc'))
            #     logits = tf.matmul(l, W_fc) + b_fc
            logits = fully_connected(inputs=l, num_outputs=self.Label_Cnt, activation_fn=None,
                                     weights_initializer=variance_scaling_initializer(),
                                     weights_regularizer=l2_regularizer(1e-4))
            return logits


        self.logits = dense_net()
        self.prob = tf.nn.softmax(logits=self.logits, name='output')
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([loss] + reg_losses, name='loss')
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits,1), tf.arg_max(self.Y,1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.Sess.run(self.logits, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x_test, y_test):
        return self.Sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False, self.dropout_rate: 1.0})

    def train(self, x_data, y_data):
        return self.Sess.run([self.accuracy, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data,
                                                                                    self.training: True, self.dropout_rate: 0.5})

    def validation(self, x_test, y_test):
        return self.Sess.run([self.loss, self.accuracy], feed_dict={self.X: x_test, self.Y: y_test, self.training: False, self.dropout_rate: 1.0})


    def parametric_relu(self, x, name):
        alphas = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.01),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg



