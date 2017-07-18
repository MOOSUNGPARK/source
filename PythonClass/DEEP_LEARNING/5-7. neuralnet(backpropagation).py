import numpy as np
from PythonClass.DEEP_LEARNING.layers import *
from PythonClass.DEEP_LEARNING.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet :
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        



