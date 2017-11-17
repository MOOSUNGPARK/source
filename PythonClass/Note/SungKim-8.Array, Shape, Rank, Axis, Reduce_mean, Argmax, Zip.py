import tensorflow as tf
import numpy as np

# ndim, shape

t = np.array([[1.,2.,3.],
             [4.,5.,6.],
             [7.,8.,9.]])

print(t.ndim) # 2
print(t.shape) # (3,3)

# shape, rank, axis

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    t = tf.constant([[[[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]],
                      [[13, 14, 15, 16],
                       [17, 18, 19, 20],
                       [21, 22, 23, 24]]]])
    print(tf.shape(t).eval()) # [1 2 3 4]

    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    print(tf.matmul(matrix1, matrix2).eval()) # 12

# random_normal, random_uniform

    print(tf.random_normal([2,3]).eval()) # [[-0.3739447  -0.32153559  0.18140531], [ 0.16315414 -0.08189275  0.05660704]]
    print(tf.random_uniform([2,3]).eval()) # [[ 0.87048829  0.52110386  0.76549602], [ 0.41504514  0.35945213  0.87130702]]

# reduce_mean, reduce_sum
    x = [[1., 2.],
         [3., 4.]]

    print(tf.reduce_mean(x, axis=0).eval()) # [2. 3.]
    print(tf.reduce_mean(x, axis=1).eval()) # [1.5 2.]

# argmax
    y = [[0., 1., 2.],
         [2., 1., 0.]]
    print(tf.argmax(y, axis=0).eval()) # [1 0 0]
    print(tf.argmax(y, axis=1).eval()) # [2 0]

# reshape, squeeze, expand_dims
    t = np.array([[[0, 1, 2],
                   [3, 4, 5]],
                  [[6, 7, 8],
                   [9, 10, 11]]])
    print(tf.reshape(t, shape=[-1,3]).eval()) # [[ 0  1  2], [ 3  4  5], [ 6  7  8], [ 9 10 11]]
    print(tf.squeeze([[0],[1],[2]]).eval()) # [0 1 2]
    print(tf.expand_dims([0, 1, 2], 1).eval()) # [[0] [1] [2]]

# casting
    print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()) # [1 2 3 4]
    print(tf.cast([True, False, 1==1, 0==1], tf.int32).eval()) # [1 0 1 0]

# ones_like, zeros_like
    z = [[0, 1, 2],
         [2, 1, 0]]
    print(tf.zeros_like(z).eval()) # [[0 0 0] [0 0 0]]
    print(tf.ones_like(z).eval())  # [[1 1 1] [1 1 1]]












