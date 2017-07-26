import numpy as np
import copy

def Relu(x):
    return np.maximum(x,0)

x = np.array([[1,2],[-1,-9],[2,-4]])
print(Relu(x))



class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        return np.maximum(x,0)

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
relulayer = Relu()
print(relulayer.forward(x))
