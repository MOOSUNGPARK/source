import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx


