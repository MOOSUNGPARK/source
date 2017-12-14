import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import *
import numpy as np

# layer_norm x / softsign / RMSprop / lr = 0.001 / epoch = 10000 / hidden_size = 256 / layers_cnt = 3

class Model:
    def __init__(self, session, in_size, out_size, name='model'):
        self.scope = name
        self.sess = session
        self.in_size = in_size
        self.hidden_size = 256
        self.out_size = out_size
        self.layers_cnt = 3
        self.lr = 0.001
        self.state_size = self.layers_cnt * 2 * self.hidden_size
        self.rnn_last_state = np.zeros((self.state_size))
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.in_size], name='X_data')
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, self.out_size], name='Y_data')
            self.Y_label = tf.reshape(self.Y, [-1, self.out_size])
            self.rnn_init_value = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name='Rnn_init_value')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')

        def add_lstm_layer(name):
            cells = []
            for idx in range(self.layers_cnt):
                cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=False,
                                         activation=tf.nn.softsign)
                if not idx == self.layers_cnt - 1 :
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_rate)
                cells.append(cell)
            lstm = rnn.MultiRNNCell(cells, state_is_tuple=False)
            outputs, state = tf.nn.dynamic_rnn(lstm, self.X, initial_state=self.rnn_init_value,
                                               dtype=tf.float32, scope=name)

            return outputs, state

        def add_lstm_layernorm_layer(name):
            cell = rnn.LayerNormBasicLSTMCell(self.hidden_size, activation=tf.nn.softsign,
                                              dropout_keep_prob=self.dropout_rate, layer_norm=True)
            lstm = rnn.MultiRNNCell([cell] * self.layers_cnt, state_is_tuple=False)
            outputs, state = tf.nn.dynamic_rnn(lstm, self.X, initial_state=self.rnn_init_value,
                                               dtype=tf.float32, scope=name)
            return outputs, state

        def add_fc_layer(name, l, hidden_size, out_size, out_layer=False):
            with tf.variable_scope(name):
                w = tf.get_variable(name=name + '_w', shape=[hidden_size, out_size], dtype=tf.float32,
                                    initializer=variance_scaling_initializer())
                b = tf.Variable(name=name + '_', initial_value=tf.random_normal([self.out_size], stddev=0.01),
                                dtype=tf.float32)
                l = tf.matmul(l, w) + b
                if not out_layer:
                    l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True)
                    l = tf.nn.elu(l, name + '_elu')
            return l

        def lstm_model():
            outputs, new_state = add_lstm_layer('LSTM')
            output_shape = tf.shape(outputs)
            outputs = tf.reshape(outputs, [-1, self.hidden_size])
            logits = add_fc_layer('FC_out', outputs, self.hidden_size, self.out_size, out_layer=True)
            final_out = tf.reshape(tf.nn.softmax(logits), [output_shape[0], output_shape[1], self.out_size])
            return logits, final_out, new_state

        self.logits, self.final_out, self.rnn_new_state = lstm_model()
        self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y_label))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, x_data, y_data):
        init_value = np.zeros((len(x_data), self.state_size))
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.X:x_data, self.Y:y_data,
                                                                        self.rnn_init_value: init_value,
                                                                        self.dropout_rate:0.5})
        return loss

    def generate(self, x_data, init_state=True):
        if init_state:
            init_value = np.zeros((self.state_size,))
        else:
            init_value = self.rnn_last_state

        out, rnn_next_state = self.sess.run([self.final_out, self.rnn_new_state],
                                            feed_dict={self.X:[x_data], self.rnn_init_value:[init_value],
                                                       self.dropout_rate:1.0})
        self.rnn_last_state = rnn_next_state[0]

        return out[0][0]






