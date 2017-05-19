# print(60*60*3 + 30 * 60)
#
# import numpy as np
#
# a= np.array([1.232412351, 12.125152125])
# print(type(a))
#
# b = np.around(a, decimals=2, out=None)
# print(b*100000)
#
#
# import math
# def cosine_similarity(v1,v2):
#     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v1)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     return sumxy/math.sqrt(sumxx*sumyy)
#
# v1,v2 = [3, 45, 7, 2], [2, 54, 13, 15]
# print(v1, v2, cosine_similarity(v1,v2))

# for i in range(12):
#     print(i, round((240/11)*i))
from librosa import time_to_samples
import numpy as np
time_to_samples(np.arrange(0.1), sr=22050)
librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)