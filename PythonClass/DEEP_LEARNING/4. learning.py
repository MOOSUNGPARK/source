import matplotlib.pylab as plt
import numpy as np

from PythonClass.DEEP_LEARNING.common.functions import softmax, cross_entropy_error
from PythonClass.DEEP_LEARNING.common.gradient import numerical_gradient
from PythonClass.DEEP_LEARNING.dataset.mnist import load_mnist


### 손실함수1. 평균 제곱 오차 ###
def mean_squard_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

### 손실함수2. 교차 엔트로피 오차 ###
def cross_entropy_error0(y, t):
    delta = 1e-7         # delta 안 넣어주면 log0 은 무한대가 되므로 에러남
    return -np.sum(t * np.log(y + delta))

### 미니 배치 ###
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]  # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 60000개 중 랜덤으로 10개 뽑기

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

### 손실함수3. 배치용 교차 엔트로피 오차(원-핫 인코딩) ###
def cross_entropy_error1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

### 손실함수4. 배치용 교차 엔트로피 오차(숫자 레이블 주어졌을 때) ###
def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

### 기울기 구하기(편미분) ###
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 같은 형상의 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def func2(x1):
    return 3.0 ** 2.0 + x1 * x1


### 소스 코드 ###
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

### 경사 감소법 ###
def gradient_descent(f, init_x, lr= 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

def func3(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])

x, x_history = gradient_descent(func3, init_x, lr=0.1, step_num=100)
plt.plot( [-5, 5], [0,0], '--b') # 가로선
plt.plot( [0,0], [-5, 5], '--b') # 세로선
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

### 신경망의 경사감소법 ###
class simplenet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화. 2행 3열의 난수 생성

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss

net = simplenet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t= np.array([0,0,1])
print(net.loss(x,t))

f = lambda w: net.loss(x,t)
dW = numerical_gradient(f, net.W)
print(dW)