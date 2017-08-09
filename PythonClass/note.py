import tensorflow as tf
import numpy as np
from dataset.mnist import load_mnist

# 데이터 정규화/ 클래스화/ 검증데이터(훈련 데이터의 10%) 추가해서 1000에폭 중 100에폭마다 검증데이터로 정확도 확인하게 /
# 배치 크기를 200으로 늘리고 그 중 랜덤으로 100개의 미니배치 뽑아서 학습시키기
# 배치 정규화

##### mnist 데이터 불러오기 및 정제 #####

############################################
# mnist 데이터 중 10000개 저장
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)
# input = np.concatenate((x_train, x_test), axis=0)
# target = np.concatenate((t_train, t_test), axis=0)
# print('input shape :', input.shape, '| target shape :', target.shape)
# a = np.concatenate((input, target), axis=1)
# np.savetxt('mnist.csv', a[:10000], delimiter=',')
############################################

# 파일 로드 및 변수 설정
mnist = np.loadtxt('mnist.csv', delimiter=',', unpack=False, dtype='float32')
print(mnist.shape)

train_num = int(mnist.shape[0] * 0.8)

x_train, x_test = mnist[:train_num,:784], mnist[train_num:,:784]
t_train, t_test = mnist[:train_num,784:], mnist[train_num:,784:]

print('x train shape :',x_train.shape, '| x target shape :',x_test.shape)
print('t train shape :',t_train.shape, '| t target shape :',t_test.shape)

global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32,[None, 784])
T = tf.placeholder(tf.float32,[None, 10])
W = tf.Variable(tf.random_uniform([784,10], -1e-7, 1e-7)) # [784,10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 어레이
b = tf.Variable(tf.random_uniform([10], -1e-7, 1e-7))    # [10] 형상을 가진 -1~1 사이의 균등분포 벡터
Y = tf.add(tf.matmul(X,W), b) # tf.matmul(X,W) + b 와 동일

############################################
# 그외 가중치 초기화 방법
# W = tf.Variable(tf.random_uniform([784,10], -1, 1)) # [784,10] 형상을 가진 -1~1 사이의 균등분포 어레이
# W = tf.get_variable(name="W", shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer()) # xavier 초기값
# W = tf.get_variable(name='W', shape=[784, 10], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 초기값
# b = tf.Variable(tf.zeros([10]))
############################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost, global_step=global_step)

############################################
# 그외 옵티마이저
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01)
############################################


##### mnist 학습시키기 #####
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())


cp = tf.train.get_checkpoint_state('./save') # save 폴더를 checkpoint로 설정
# checkpoint가 설정되고, 폴더가 실제로 존재하는 경우 restore 메소드로 변수, 학습 정보 불러오기
if cp and tf.train.checkpoint_exists(cp.model_checkpoint_path):
    saver.restore(sess, cp.model_checkpoint_path)
    print(sess.run(global_step),'회 학습한 데이터 로드 완료')
# 그렇지 않은 경우 일반적인 sess.run()으로 tensorflow 실행
else:
    sess.run(tf.global_variables_initializer())
    print('새로운 학습 시작')

# epoch, batch 설정
epoch = 1000
batch_size = 100
total_batch = int(x_train.shape[0] / batch_size)

# 설정한 epoch 만큼 루프
for each_epoch in range(epoch):
    total_cost = 0

    # 각 epoch 마다 batch 크기만큼 데이터를 뽑아서 학습
    for idx in range(0, total_batch, batch_size):
        batch_x, batch_y = x_train[idx : idx+batch_size], t_train[idx : idx+batch_size]

        _, cost_val = sess.run([optimizer, cost], feed_dict={X : batch_x, T : batch_y})
        total_cost += cost_val

    print('Epoch:', '%04d' % (each_epoch + 1),
          'Avg. cost =', '{:.8f}'.format(total_cost / total_batch),
          )

print('최적화 완료!')

# 최적화가 끝난 뒤, 변수와 학습 정보 저장
saver.save(sess, './save/mnist_dnn.ckpt', global_step=global_step)

##### 학습 결과 확인 #####
# equal 메소드로 (True, False, True, ....) 형식으로 출력
correct_prediction = tf.equal(tf.argmax(T, 1), tf.argmax(Y, 1))
# tf.cast로 True, False 등을 float32 형태로 변경 True -> 1.0, False -> 0.0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Train 정확도 :', sess.run(accuracy, feed_dict={X: x_train, T: t_train}))
print('Test 정확도:', sess.run(accuracy, feed_dict={X: x_test, T: t_test}))