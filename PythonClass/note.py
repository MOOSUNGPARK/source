import numpy as np

class Affine:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(self.x,self.w) + self.b

    def backward(self, x, out):
        self.x = x
        dx = np.dot(out, self.w.T)
        dw = np.dot(self.x.T, out)
        db = np.sum(out, axis=0)
        return dx, dw, db

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


x = np.array([1,2], ndmin=2)
w1 = np.array([[1,3,5],[2,4,6]])
b1 = np.array([1,2,3])
w2 = np.array([[1,4],[2,5],[3,6]])
b2 = np.array([1,2])

affine1 = Affine(w1,b1)
relu = Relu()
affine2 = Affine(w2,b2)
h = affine1.forward(x)
h2 = relu.forward(h)
out = affine2.forward(h2)

print(out)

dx2, dw2, db2 = affine2.backward(h2, out)
dh2 = relu.backward(dx2)
dx1, dw1, db1 = affine1.backward(x, dh2)
print(dx1, dw1, db1)