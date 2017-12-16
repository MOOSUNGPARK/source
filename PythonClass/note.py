import tensorflow as tf
import numpy as np
# sample = " if you want you"
# idx2char = list(set(sample))  # index -> char
# char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex
#
# print(idx2char)

sequence = [[[1,0,0,0],
             [0,1,0,0],
             [1,1,0,0],
             [0,0,0,0],
             [0,0,0,0]],
            [[3,3,3,3],
             [2,2,1,1],
             [1,0,0,1],
             [2,1,-1,0],
             [0,1,0,0]]]
used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
length = tf.reduce_sum(used, 1)
length = tf.cast(length, tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # leng = sess.run(length)
    # use = sess.run(used)
    # print(leng)
    # print(use)
    shape = sess.run(tf.shape(sequence))
    reshape = sess.run(  tf.reshape(tf.reshape(sequence, [-1, 4]), [2,5,4]  ))
    flat = sess.run(tf.reshape(sequence, [-1, 4]))
    print(flat)
    print(reshape)