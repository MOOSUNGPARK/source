import sys, os
# sys.path.append('C:\\python\\source\\PythonClass\\DEEP_LEARNING\\dataset')
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle

### 이미지 복원 ###
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = x_train[0]
# print(label)

img = img.reshape(28,28)
# print(img.shape)
# img_show(img)


### mnist 신경망 ###
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    return exp_a / np.sum(exp_a)

def get_data():
    _, (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open('C:\\python\\deep-learning-from-scratch\\ch03\\sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0

for i in range(len(x)) :
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
    if p == t[i] :
        accuracy_cnt += 1
print('Accuracy : ', str(float(accuracy_cnt)/len(x)))


### 배치 처리 ###
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # axix = 1 은 행을 축으로 각 행마다 argmax 찾는다는 소리
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print('배치처리 Accuracy : ',accuracy_cnt/len(x))

