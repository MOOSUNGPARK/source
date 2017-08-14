import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os


# 신경망 구현
class Qnetwork():
    def __init__(self, h_size):
        # 신경망은 게임으로부터 벡터화된 배열로 프레임을 받아서
        # 이것을 리사이즈 하고, 4개의 콘볼루션 레이어를 통해 처리한다.

        # 입력값을 받는 부분 21168 차원은 84*84*3 의 차원이다.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        # conv2d 처리를 위해 84x84x3 으로 다시 리사이즈
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])

        # 첫번째 콘볼루션은 8x8 커널을 4 스트라이드로 32개의 activation map을 만든다
        # 출력 크기는 (image 크기 - 필터 크기) / 스트라이드 + 1 이다.
        # zero padding이 없는 VALID 옵션이기 때문에
        # (84-8)/4 + 1
        # 20x20x32 의 activation volumn이 나온다
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        # 두번째 콘볼루션은 4x4 커널을 2 스트라이드로 64개의 activation map을 만든다.
        # 출력 크기는 9x9x64
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        # 세번째 콘볼루션은 3x3 커널을 1 스트라이드로 64개의 activation map을 만든다.
        # 출력 크기는 7x7x64
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        # 네번째 콘볼루션은 7x7 커널을 1 스트라이드 512개의 activation map을 만든다.
        # 출력 크기는 1x1x512
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)

        # 마지막 콘볼루션 레이어의 출력을 가지고 2로 나눈다.
        # streamAC, streamVC 는 각각 1x1x256
        self.streamAC, self.streamVC = tf.split(3, 2, self.conv4)
        # 이를 벡터화한다. streamA 와 streamV는 256 차원씩이다.
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        # 256개의 노드를 곱해서 각각 A와 V를 구하는 가중치
        self.AW = tf.Variable(tf.random_normal([256, env.actions]))
        self.VW = tf.Variable(tf.random_normal([256, 1]))
        # 점수화 한다.
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # 가치 함수 값에 이득에서 이득의 평균을 빼준 값들을 더해준다.
        self.Qout = self.Value + tf.sub(self.Advantage,
                                        tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        # 이것으로 행동을 고른다.
        self.predict = tf.argmax(self.Qout, 1)

        # 타겟과 예측 Q value 사이의 차이의 제곱합이 손실이다.
        # 타겟Q를 받는 부분
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        # 행동을 받는 부분
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        # 행동을 one_hot 인코딩 하는 부분 (tf.one_hot은 내 컴퓨터에서 GPU 에러를 내기에 다음의 해법을 찾아 적용)
        def one_hot_patch(x, depth):
            sparse_labels = tf.reshape(x, [-1, 1])
            derived_size = tf.shape(sparse_labels)[0]
            indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
            concated = tf.concat(1, [indices, sparse_labels])
            outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(depth, [1])])
            return tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

        self.actions_onehot = one_hot_patch(self.actions, env.actions)

        # 각 네트워크의 행동의 Q 값을 골라내는 것
        # action 번째를 뽑고 싶지만 tensor는 인덱스로 쓸 수 없어서 이렇게 하는듯(내 생각)
        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)

        # 각각의 차이
        self.td_error = tf.square(self.targetQ - self.Q)
        # 손실
        self.loss = tf.reduce_mean(self.td_error)
        # 최적화 방법 adam
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # 업데이트 함수
        self.updateModel = self.trainer.minimize(self.loss)
