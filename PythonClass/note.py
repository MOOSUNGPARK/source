import numpy as np

def NAND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    res = x1 * w1 + x2 * w2

    if res >= theta :
        return 0
    else :
        return 1

print(NAND(0,0))

def OR(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = 0.3
    res = np.sum(x*w)

    if res >= theta :
        return 1
    else :
        return 0

print(OR(1,1))

def AND(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = 0.7
    res = np.sum(x*w)

    if res >= theta :
        return 1
    else :
        return 0

def XOR(x1, x2) :
    x_nand = NAND(x1, x2)
    x_or =OR(x1, x2)
    return AND(x_nand, x_or)

print(XOR(0,0))

