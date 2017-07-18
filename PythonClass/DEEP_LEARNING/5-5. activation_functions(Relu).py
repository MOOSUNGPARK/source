import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

if __name__ == '__main__':
    relu = Relu()
    x = np.array([[1.0, -0.5], [-0.2, 3.0]])


