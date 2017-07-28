import numpy as np

# 차원 명시해줘야 에러가 안남
x = np.array([[1, 2],[2,4]])
w = np.array([[1, 3, 5], [2, 4, 6]])
b = np.array([1, 2, 3])


class Affine:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.w) + self.b

    def backward(self, x, out):
        self.x = x
        dx = np.dot(out, self.w.T)
        dw = np.dot(self.x.T, out)
        db = np.sum(out, axis=0)
        return dx, dw, db


affine1 = Affine(w, b)
out = affine1.forward(x)
print(out)
print(affine1.backward(x, out))



