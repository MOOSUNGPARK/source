# coding: utf-8
import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class SimpleConvNet:

    def __init__(self, input_dim=(1,28,28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = ((input_size - filter_size + 2*filter_pad) / filter_stride) + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.lastlayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
#
# # 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
#
#
# # 하이퍼파라미터
# iters_num = 10000  # 반복 횟수를 적절히 설정한다.
# train_size = x_train.shape[0] # 60000 개
# batch_size = 100  # 미니배치 크기
# learning_rate = 0.1
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
#
#
# # 1에폭당 반복 수
# iter_per_epoch = max(train_size / batch_size, 1)
# print(iter_per_epoch) # 600
#
# for i in range(iters_num): # 10000
#     # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
#     batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#
#
#     # 기울기 계산
#     #grad = network.numerical_gradient(x_batch, t_batch)
#     grad = network.gradient(x_batch, t_batch)
#     # 매개변수 갱신
#
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
#
#
#     # 학습 경과 기록
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
#     # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크
#
#     if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
#         print(x_train.shape) # 60000,784
#         train_acc = network.accuracy(x_train, t_train)
#         test_acc = network.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
#         test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
#         print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
#
#
# # 그래프 그리기
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()