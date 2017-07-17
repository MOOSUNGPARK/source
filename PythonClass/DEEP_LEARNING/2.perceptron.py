import numpy as np

### 가중치 반영 전 ###
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    res = w1*x1 + w2*x2
    if res >= theta:
        return 1
    else:
        return 0

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print('AND',np.sum(w*x) + b)

### 가중치와 편향 반영 후 ###
def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    res = np.sum(w*x) + b
    if res >= 0:
        return 1
    else :
        return 0
print('AND2',AND2(0,1))

### NAND ###
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    res = np.sum(w*x) + b
    if res >= 0:
        return 1
    else :
        return 0
print('NAND',NAND(0,1))

### OR ###
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3  # AND 와 bias 만 다름
    res = np.sum(w*x) + b
    if res >= 0 :
        return 1
    else :
        return 0
print('OR', OR(0,1))

### XOR ###
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
print('XOR',XOR(1,1))






