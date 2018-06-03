import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# 1. 파일 목록 지정
filename_queue = tf.train.string_input_producer(['C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test_data_1.csv'], shuffle=False, name='filename_queue',
                                                num_epochs=None)
print(filename_queue)
# 2. Reader 정의
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
print(value)
# 3. decode & data type 지정
record_defaults = [[0.] for _ in range(126 * 126 + 1)] # float으로 지정
xy = tf.decode_csv(value, record_defaults=record_defaults) # csv


print(xy)



# 4. batch를 통해 가져와 역할에 맞게 배분 (6개씩 가져옴)
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=100)

# shape 주의!
X = tf.placeholder(tf.float32, shape=[None, 6])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([6, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# 세션 시작
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 멀티쓰레드가 함께 종료 되도록 도움
coord = tf.train.Coordinator() #####################
# 동일한 큐 안에 tensor가 동작하도록 쓰레드 생성에 도움
threads = tf.train.start_queue_runners(sess=sess, coord=coord)######################################

for step in range(10):
    # 데이터를 배치로 가져옴
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    print(y_batch)
#     # 5. 학습
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y:y_batch})
#     if step % 1000 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
#         # 2000
#         # Cost: 5.89609
#         # Prediction:
#         # [[155.44619751]
#         #  [182.85728455]
#         #  [182.59645081]
#         #  [195.05700684]
#         #  [141.46882629]
#         #  [97.68396759]]
#
# # 쓰레드 멈춤
coord.request_stop() ########################################
# # 쓰레드가 끝나기 전에 프로그램이 종료되는 것을 막기 위해 기다림
coord.join(threads) #################################################


# (len(data) / batch) * epoch