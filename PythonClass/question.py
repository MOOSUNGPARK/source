import numpy as np
from collections import OrderedDict
from common.layers import *
from common.optimizer import Adam

class NeuralNet :
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = np.sqrt(2) * np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(2) * np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.sqrt(2) * np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastlayers = IdentityWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayers.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = self.lastlayers.backward()
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads


load_ballloc = 'c:\python\data\pingpong_move.csv'

x = np.genfromtxt(load_ballloc, delimiter=',')[:,:4]
t = np.genfromtxt(load_ballloc, delimiter=',')[:,4]
neural = NeuralNet(input_size=4, hidden_size=10, output_size=1)

print(neural.predict(x), t)
optimizer = Adam(lr = 0.01)

for i in range(30000):
    grads = neural.gradient(x,t)
    params = neural.params
    optimizer.update(params, grads)
    # for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
    #     neural.params[key] -= 0.01 * grads[key]
    if i % 100 == 0 :
        print(i)
        print(neural.loss(x, t))
    # print(neural.loss(x,t))

print(neural.predict(x), t)