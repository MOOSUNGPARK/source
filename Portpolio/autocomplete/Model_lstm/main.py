import tensorflow as tf
import numpy as np
import time
import sys
import Portpolio.autocomplete.Model_lstm.util as Util
from Portpolio.autocomplete.Model_lstm.lstm_practice import Model

ckpt_file = ""
TEST_PREFIX = "제1조"  # Prefix to prompt the network in test mode

print("Usage:")
print('\t\t ', sys.argv[0], ' [ckpt model to load] [prefix, e.g., "The "]')
if len(sys.argv) >= 2:
    ckpt_file = sys.argv[1]
if len(sys.argv) == 3:
    TEST_PREFIX = sys.argv[2]

########################
loc = 'C:\\python\\source\\Portpolio\\autocomplete\\data\\대한민국헌법.txt'

TRAIN = True
txt, vocab, vocab_to_char = Util.load_txt_vocab(loc=loc)
data = Util.txt_one_hot(txt, vocab)
input_size = class_size = len(vocab)
batch_size = 64  # 128
time_steps = 100  # 50
shape = [batch_size, time_steps, input_size]
possible_batch_range = range(data.shape[0] - time_steps - 1)
epochs = 3000  # 20000
Generated_sentence_length = 1000  # Number of test characters of text to generate after training the network

# Initialize the network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.InteractiveSession(config=config)

with tf.Session() as sess:
    model = Model(session=sess, in_size=input_size, out_size=class_size, name='LSTM_Model')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if ckpt_file == '' or TRAIN:
        stime = time.time()

        for epoch in range(epochs):
            sstime = time.time()

            x_batch, y_batch = Util.make_batch(data, possible_batch_range, shape)
            loss = model.train(x_batch, y_batch)

            if epoch % 100 == 0 :
                eetime = time.time()
                print('Epoch :', epoch, '\t\tLoss :', loss, '\t\tConsumption Time :', round( (eetime - sstime) * 100, 6))

        print('Learning Finished!')
        etime = time.time()
        print('Total training Time :', round(etime - stime, 6))

        saver.save(sess, 'C:\\python\\source\\Portpolio\\autocomplete\\log\\model.ckpt')

    else:
        saver.restore(sess, ckpt_file)

    TEST_PREFIX = TEST_PREFIX.lower()

    for idx in range(len(TEST_PREFIX)):
        out = model.generate(Util.txt_one_hot(TEST_PREFIX[idx], vocab), idx==0)

    Generated_sentence = TEST_PREFIX

    for idx in range(Generated_sentence_length):
        element = np.random.choice(range(len(vocab)), p=out)
        Generated_sentence += vocab_to_char[element]
        out = model.generate(Util.txt_one_hot(vocab_to_char[element], vocab), False)

    print('Generated Sentence :\n', Generated_sentence)

