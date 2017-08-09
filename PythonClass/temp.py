import tensorflow as tf
import numpy as np

a = tf.placeholder("float")  # 공간을 만든다.
b = tf.placeholder("float")
y = tf.multiply(a, b)  # 곱을 한다.
with tf.Session() as sess:
    print(sess.run(y, feed_dict={a: 10, b: 32}))

