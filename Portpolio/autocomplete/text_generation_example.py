import numpy as np
import random
import sys
import tensorflow as tf

path = 'C:\\python\\source\\Portpolio\\autocomplete\\data\\nietzsche_example.txt'
text = open(path).read().lower()
print('text',text)
print('len(text)', len(text))

chars = sorted(list(set(text)))
print('chars',chars)
print('len(chars):', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print('char_indices', char_indices)
print('indices_char', indices_char)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('sentences', sentences)
print('next_chars', next_chars)
print('len(sentences)', len(sentences))

n_char=len(chars)

print('Vectorization...')
X_data = np.zeros((len(sentences), maxlen, n_char), dtype=np.int32)
Y_data = np.zeros((len(sentences), n_char), dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X_data[i, t, char_indices[char]] = 1
        # print('char_indices[char]', char_indices[char])
        # print('len(X_data)', len(X_data))
        # print('len(X_data[0]', len(X_data[0]), X_data[0])
        # print('len(X_data[0][0]', len(X_data[0][0]), X_data[0][0])

        # print('\nX_data',X_data)
    Y_data[i, char_indices[next_chars[i]]] = 1
    # print('\nY_data', Y_data)

print ("pre-processing ready")


# Parameters
learning_rate = 0.01
training_iters = 0
batch_size = 128
display_step = 10
n_hidden = 128

# tf Graph input
x = tf.placeholder("float", [None, maxlen, n_char])
y = tf.placeholder("float", [None, n_char])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_char]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_char]))
}
print ("parameters ready")

with tf.variable_scope("model"):
    # tf.get_variable_scope().reuse_variables()

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    # x_t = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    # x_t = tf.reshape(x_t, [-1, n_char])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x_t = tf.split(0, maxlen, x_t)

    # Define a lstm cell with tensorflow
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    # Linear activation, using rnn inner loop last output
    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

# pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("Network ready")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def make_batches(size, batch_size):
    nb_batch = int(np.floor(size / float(batch_size)))
    # nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]


print("functions ready")


import itertools

ins=[X_data,Y_data]

n_train=X_data.shape[0]

index_array = np.arange(n_train)

np.random.shuffle(index_array)

batches = make_batches(n_train, batch_size)

ins=[slice_X(ins,index_array[batch_start:batch_end]) for batch_start, batch_end in batches]
print('ins', ins)
iterator=itertools.cycle((data for data in ins if data != []))

print ("datasets ready")

sample_step = 1000
# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    init = tf.global_variables_initializer()

    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        [batch_x, batch_y] = next(iterator)
        # Run optimization op (backprop)
        _, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

        start_index = random.randint(0, len(text) - maxlen - 1)

        if step % sample_step == 0:
            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(200):
                x_sample_input = np.zeros((1, maxlen, n_char))
                for t, char in enumerate(sentence):
                    x_sample_input[0, t, char_indices[char]] = 1.

                preds = sess.run(pred, feed_dict={x: x_sample_input})
                next_index = np.argmax(preds)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    print("Optimization Finished!")