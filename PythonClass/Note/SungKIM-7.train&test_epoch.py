import tensorflow as tf
import numpy as np

xy = np.loadtxt('c:\\python\\data\\Animal.csv', delimiter=',', dtype=np.float32)
x_train = xy[:80,0:-1]
x_test = xy[80:,0:-1]
y_train = xy[:80,[-1]]
y_test = xy[80:,[-1]]
nclasses = 7

X = tf.placeholder(tf.float32, shape=[None,16])
Y = tf.placeholder(tf.int32, shape=[None,1])
Y_one_hot = tf.one_hot(Y, nclasses)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nclasses])

W = tf.Variable(tf.random_normal([16, nclasses]), name='weight')
b = tf.Variable(tf.random_normal([nclasses]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(500):
        cost_val, _, _ = sess.run([cost, W, optimizer],
                                      feed_dict={X:x_train, Y:y_train})
        print(step, cost_val)

    pred = sess.run(prediction, feed_dict={X: x_test})

    for p, y in zip(pred, y_test.flatten()):
        print('[{}] Prediction : {}, Label : {}'.format(p==int(y), p, int(y)))

    print('Accuracy: ', sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))


# MinMaxSacler : 0~1 사이 값으로 nomarlization





