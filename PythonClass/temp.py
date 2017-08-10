import tensorflow as tf
import numpy as np

correct_prediction = tf.zeros([5,3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(correct_prediction.get_shape())
