import tensorflow as tf

class Qnetwork():
    def __init__(self, h_size, number_of_actions=3):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 7056], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 1])
        self.conv1 = tf.contrib.layers.convolution2d(
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d(
            inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(3, 2, self.conv4)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size / 2, number_of_actions]))
        self.VW = tf.Variable(tf.random_normal([h_size / 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.substr(self.Advantage, tf.reduce_mean(self.Advantage, 1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, number_of_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.matmul(self.Qout, self.actions_onehot), reduction_indices=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
