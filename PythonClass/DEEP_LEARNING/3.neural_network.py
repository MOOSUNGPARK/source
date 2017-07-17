import numpy as np
import matplotlib.pylab as plt

### 계단 함수 ###

def step_function0(x):
    if x >= 0 :
        return 1
    else :
        return 0

### 계단 함수(array 형태도 받을 수 있게) ###

def step_function1(x):
    y = x > 0   # np.array 형태의 x 에서 0보다 크면 true, 아니면 false를 출력
                # 결과 예 : array([False, True, True], dtype = bool)
    return y.astype(np.int) # 원하는 자료형(np.int)으로 변경(파이썬은 False 는 0으로, True 는 1로 출력해줌)

print('step_function1', step_function1(np.array([0,1,2,3,-5])))

### 최종 계단 함수 ###
def step_function(x):
    return np.array(x > 0, dtype = np.int) ### 꼭 알아두기!!! ###

print('step_function', step_function(np.array([0,1,2,3,-5])))

### 계단 함수 그래프 ###
'''
x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)  # y축 범위 지정
plt.show()
'''

### 시그모이드 함수 ###
def sigmoid(x):
    return 1/(1 + np.exp(-x))

x = np.array([-1, 1, 2])
print('sigmoid',sigmoid(x))

### 시그모이드 그래프 ###
'''
x = np.arange(-5, 5, 0.1)
plt.plot(x, sigmoid(x))
plt.ylim(-0.1, 1.1)
plt.show()
'''

### 렐루 함수 ###
def relu(x):
    return np.maximum(0,x)  # np.max 는 input이 하나임.

print('relu',relu(1))

### 다차원 배열 ###
a = np.array([ [1,2],
               [3,4],
               [5,6] ])
print('a\n',a)
print('차원수', np.ndim(a))
print('행렬 형태', np.shape(a))

### 행렬의 내적(행렬곱) ###
A = np.array([ [1,2],
               [3,4]])
print('A의 형태', A.shape)
B = np.array([ [5,6],
               [7,8]])
print('B의 형태', B.shape)
print('dot\n',np.dot(A,B))

### 신경망의 내적 ###
X = np.array([1,2])
print('X shape', X.shape)
W = np.array([ [1,3,5],
               [2,4,6] ])
print('W shape', W.shape)
print('X\n',X,'\nW\n',W,'\ndot\n',np.dot(X,W))

### 3층 신경망 ###
def identity_function0(x): # 출력층의 활성화함수로 이 경우 항등함수임
                          # (회귀시 항등함수, 분류 시 소프트맥스 함수 주로 씀)
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function0(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print('neural net', y)

### 항등 함수 ###
def identity_function(x):
    return x

### 소프트맥스 함수 ###
def softmax_function0(a): # 소프트맥스 함수
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)

print('softmax',softmax_function0(np.array([0.3,2.9,4.0])))

### 소프트맥스 함수 개선(너무 큰 값이 나와서 에러 뜨지 않게 하는 방법) ###
def softmax_function(a):
    c = np.max(a)       # 큰 값이 나오지 않도록 a 배열의 최대값을 빼줌. 그래도 결과는 같다. p.93
    exp_a = np.exp(a-c)
    return exp_a / np.sum(exp_a)




