import tensorflow as tf

class Model:
    def __init__(self, sess, name, label_cnt):
        self.sess = sess
        self.name = name
        self.label_cnt = label_cnt
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')
                self.X = tf.placeholder(tf.float32, [None, 48 * 48], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 48, 48, 1])
                self.Y = tf.placeholder(tf.float32, [None, self.label_cnt], name='y_data')

            with tf.name_scope('conv_layer1') as scope: # [N, 48, 48, 1] -> [N, 24, 24, 32]
                self.W1 = tf.get_variable(name='W1', shape=[3,3,1,32], dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1 = tf.nn.conv2d(input=X_img, filter=self.W1, strides=[1,1,1,1], padding='SAME')
                self.L1 = self.BN(input=self.L1, scale=True, training=self.training, name='Conv1_BN')
                self.L1 = self.parametric_relu(self.L1,'R1')
                self.L1 = tf.nn.max_pool(value=self.L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            with tf.name_scope('conv_layer2') as scope: # [N, 24, 24, 32] -> [N, 12, 12, 64]
                self.W2 = tf.get_variable(name='W2', shape=[3,3,32,64], dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2 = tf.nn.conv2d(input=self.L1, filter=self.W2, strides=[1,1,1,1], padding='SAME')
                self.L2 = self.BN(input=self.L2, scale=True, training=self.training, name='Conv2_BN')
                self.L2 = self.parametric_relu(self.L2,'R2')
                self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            with tf.name_scope('conv_layer3') as scope: # [N, 12, 12, 64] -> [N, 6, 6, 128]
                self.W3 = tf.get_variable(name='W3', shape=[3,3,64,128], dtype=tf.float32,
                                           initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3 = tf.nn.conv2d(input=self.L2, filter=self.W3, strides=[1,1,1,1], padding='SAME')
                self.L3 = self.BN(input=self.L3, scale=True, training=self.training, name='Conv3_BN')
                self.L3 = self.parametric_relu(self.L3,'R3')
                self.L3 = tf.nn.max_pool(value=self.L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            with tf.name_scope('conv_layer4') as scope: # [N, 6, 6, 128] -> [N, 3, 3, 256]
                self.W4 = tf.get_variable(name='W4', shape=[3,3,128,256], dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1,1,1,1], padding='SAME')
                self.L4 = self.BN(input=self.L4, scale=True, training=self.training, name='Conv4_BN')
                self.L4 = self.parametric_relu(self.L4,'R4')
                self.L4 = tf.nn.max_pool(value=self.L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                self.L4 = tf.reshape(self.L4, shape=[-1, 3 * 3 * 256])

            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[3 * 3 * 256, 1000], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L5 = tf.matmul(self.L4, self.W_fc1) + self.b_fc1
                self.L5 = self.BN(input=self.L5, scale=True, training=self.training, name='Fc1_BN')
                self.L_fc1 = self.parametric_relu(self.L5, 'R5')

            with tf.name_scope('fc_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[1000, 1000], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc2'))
                self.L6 = tf.matmul(self.L_fc1, self.W_fc2) + self.b_fc2
                self.L6 = self.BN(input=self.L6, scale=True, training=self.training, name='Fc2_BN')
                self.L_fc2 = self.parametric_relu(self.L6, 'R6')

            self.W_out = tf.get_variable(name='W_out', shape=[1000,self.label_cnt], dtype=tf.float32,
                                         initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[self.label_cnt], name='b_out'))
            self.logits = tf.matmul(self.L_fc2, self.W_out) + self.b_out

        self.L2_reg = 0.01/(2*tf.to_float(tf.shape(self.Y[0]))) * tf.reduce_sum(tf.square(self.W_out))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + self.L2_reg
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits,1), tf.arg_max(self.Y, 1)), dtype=tf.float32))


    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def BN(self, input, training, scale, name, decay=0.99):
        return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training,
                                            updates_collections=None, scope=name)
    def parametric_relu(self, x, name):
        alphas = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.01),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg



