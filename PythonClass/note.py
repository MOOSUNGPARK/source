import sys, os
# sys.path.append('C:\\python\\source\\PythonClass\\DEEP_LEARNING\\dataset')
from dataset.mnist import load_mnist
import numpy as np


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    return exp_a/ np.sum(exp_a)


a = np.array([[[1,3,2],
               [1,2,3],
               [4,5,6]] ,

              [[1,2,3],
               [7,8,9],
               [10,11,12]]
              ])


print(   np.argmax(softmax(a), axis=0)  )
print(a.shape)
