# 알파고를 만든 구글의 딥마인드의 논문을 참고한 DQN 모델을 생성합니다.
# http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
import tensorflow as tf
import numpy as np
import random
from collections import deque

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)