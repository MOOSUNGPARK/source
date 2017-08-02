# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from dataset.mnist import load_mnist
from collections import OrderedDict
import matplotlib.pyplot as plt

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.4):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] =weight_init_std * np.random.randn(input_size, hidden_size)  # 표준 정규 분포를 따르는 난수 생성
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)
        # 계층 생성
        self.layers = OrderedDict()  # forward, backward 시 계층 순서대로 수행하기 위해 순서가 있는 OrderedDict 를 사용.
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Batchnorm1'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Relu1'] = Relu()
        # self.layers['Dropout1'] = Dropout()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Batchnorm2'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Relu2'] = Relu()
        # self.layers['Dropout2'] = Dropout()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Batchnorm3'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Relu3'] = Relu()
        # self.layers['Dropout3'] = Dropout()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Batchnorm4'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Relu4'] = Relu()
        # self.layers['Dropout4'] = Dropout()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.lastLayer = SoftmaxWithLoss()

        # L2 정규화
        self.weight_decay_lambda = 0.001

    def predict(self, x):
        for layer in self.layers.values():  # Affine1 -> Relu1 -> Affine2
            x = layer.forward(x)  # 각 계층마다 forward 수행
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):  # x : (100, 1024), t : (100, 10)
        y = self.predict(x)  # (100, 10) : 마지막 출력층을 통과한 신경망이 예측한 값
        weight_decay = 0
        for idx in range(1, 6):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        return self.lastLayer.forward(y, t) + weight_decay # 마지막 계층인 SoftmaxWithLoss 계층에 대해 forward 수행

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # [[0.1, 0.05, 0.5, 0.05, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1], ....] -> [2, 4, 2, 1, 9, ....]
        if t.ndim != 1: t = np.argmax(t, axis=1)  # t.ndim != 1 이면 one-hot encoding 인 경우이므로, 2차원 배열로 값이 들어온다
        accuracy = np.mean(y == t)
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()  # 역전파를 수행하기 위해 기존 layer 순서를 반대로 바꾼다.

        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        for idx in range(1,6) :
            grads['W'+str(idx)] = self.layers['Affine'+str(idx)].dW + self.weight_decay_lambda * self.layers['Affine'+str(idx)].W
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db
        return grads



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]  # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch)  # 600

for i in range(iters_num):  # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size)  # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5'):
        network.params[key] -= learning_rate * grad[key]
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)  # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0:  # 600 번마다 정확도 쌓는다.
        print(x_train.shape)  # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)  # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()