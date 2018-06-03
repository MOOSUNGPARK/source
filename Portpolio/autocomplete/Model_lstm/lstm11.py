import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import *
import numpy as np

# variational dropout o 0.8 / layer_norm o / elu / rmsprop / lr = 0.003 / epoch = 10 / hidden_size = 512 / layers_cnt = 3

class Model:
    def __init__(self, session, in_size, out_size, name='model'):
        self.scope = name
        self.sess = session
        self.in_size = in_size
        self.hidden_size = 256
        self.out_size = out_size
        self.output_shape = None
        self.layers_cnt = 2
        self.lr = 0.001
        self.rnn_last_state = None
        self.dtype = tf.float32
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(dtype=self.dtype, shape=[None, None, self.in_size], name='X_data')
            self.Y = tf.placeholder(dtype=self.dtype, shape=[None, None, self.out_size], name='Y_data')
            self.Y_label = tf.reshape(self.Y, [-1, self.out_size])
            self.dropout_rate = tf.placeholder(dtype=self.dtype, name='dropout_rate')

        #dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
        #multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
        # multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

        def add_lstm_layer(name):
            cells = [rnn.LSTMCell(self.hidden_size, activation=tf.nn.softsign)]
            dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate) for cell in cells]
            multicell = rnn.MultiRNNCell(dropcells)
            multicell = rnn.DropoutWrapper(multicell, output_keep_prob=self.dropout_rate)

            outputs, state = tf.nn.dynamic_rnn(multicell, self.X, initial_state=self.rnn_last_state,
                                               dtype=self.dtype, scope=name)

            return outputs, state

        def add_layernorm_lstm_layer(name):
            cells = [rnn.LayerNormBasicLSTMCell(self.hidden_size, activation=tf.nn.softsign,
                                                dropout_keep_prob=self.dropout_rate, layer_norm=True)]
            multicell = rnn.MultiRNNCell(cells)
            multicell = rnn.DropoutWrapper(multicell, output_keep_prob=self.dropout_rate)
            outputs, state = tf.nn.dynamic_rnn(multicell, self.X, dtype=self.dtype,
                                               initial_state=self.rnn_last_state, scope=name)
            return outputs, state

        def add_variational_lstm_layer(name):
            cells = []
            for idx in range(self.layers_cnt):
                cell = rnn.BasicLSTMCell(self.hidden_size, activation=tf.nn.softsign)
                cell = rnn.DropoutWrapper(cell, dtype=self.X.dtype,
                                                input_keep_prob=self.dropout_rate,
                                                state_keep_prob=self.dropout_rate,
                                                output_keep_prob=1.0,
                                                variational_recurrent=True,
                                                input_size=self.in_size if idx == 0 else tf.TensorShape(self.hidden_size))
                cells.append(cell)

            multicell = rnn.MultiRNNCell(cells)
            outputs, state = tf.nn.dynamic_rnn(multicell, self.X, initial_state=self.rnn_last_state,
                                               dtype=self.X.dtype, scope=name)
            return outputs, state

        def add_variational_layernorm_lstm_layer(name):
            cells = []
            for idx in range(self.layers_cnt):
                cell = rnn.LayerNormBasicLSTMCell(self.hidden_size, activation=tf.nn.softsign, layer_norm=True)
                cell = rnn.DropoutWrapper(cell, dtype=self.X.dtype,
                                                input_keep_prob=self.dropout_rate,
                                                state_keep_prob=self.dropout_rate,
                                                output_keep_prob=1.0,
                                                variational_recurrent=True,
                                                input_size=self.in_size if idx == 0 else tf.TensorShape(self.hidden_size))
                cells.append(cell)

            multicell = rnn.MultiRNNCell(cells)
            outputs, state = tf.nn.dynamic_rnn(multicell, self.X, dtype=self.dtype,
                                               initial_state=self.rnn_last_state, scope=name)
            return outputs, state

        def add_fclayer(name, l, num_input, num_output, out_layer=False):
            with tf.variable_scope(name):
                w = tf.get_variable(name=name + '_w', shape=[num_input, num_output], dtype=self.dtype,
                                    initializer=variance_scaling_initializer())
                b = tf.Variable(tf.constant(value=0.001, shape=[num_output], name=name + '_b'))
                l = tf.matmul(l, w) + b
                if not out_layer:
                    l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True)
                    l = tf.nn.elu(l, name + '_elu')
            return l

        def lstm_model():
            outputs, new_state = add_lstm_layer('LSTM')
            self.output_shape = tf.shape(outputs)
            outputs = tf.reshape(outputs, [-1, self.hidden_size])
            logits = linear(outputs, self.out_size)
            final_out = tf.reshape(tf.nn.softmax(logits), [self.output_shape[0], self.output_shape[1], self.out_size])
            return logits, final_out, new_state

        self.logits, self.final_out, self.rnn_new_state = lstm_model()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y_label)
        self.loss = tf.reshape(loss, [self.output_shape[0], -1])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, x_data, y_data):
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.X:x_data, self.Y:y_data,
                                                                        self.dropout_rate:0.8})
        return loss

    def generate(self, x_data):
        out, rnn_next_state = self.sess.run([self.final_out, self.rnn_new_state],
                                            feed_dict={self.X:[x_data], self.dropout_rate:1.0})
        self.rnn_last_state = rnn_next_state

        return out[0][0]






