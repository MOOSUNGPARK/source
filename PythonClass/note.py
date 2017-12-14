import numpy as np
# sample = " if you want you"
# idx2char = list(set(sample))  # index -> char
# char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex
#
# print(idx2char)

import random

# a = range(100)
# print(random.sample(a, 10))
print(np.random.choice(range(5), p=[0,0,0.5,0.5,0]))
