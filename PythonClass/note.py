import numpy as np

a = np.array([ [1,2], [3,4] ])
b = np.array([ [5,6], [7,8] ])
print(a*b)

aa = np.matrix([ [1,2], [3,4] ])
bb = np.matrix([ [5,6], [7,8] ])
print(aa*bb)

c = [ [1,2], [3,4] ]
d = [ [5,6], [7,8] ]
e = [ [0,0], [0,0] ]

for rn in range(len(c)):
    for cn in range(len(c[0])):
        e[rn][cn] = c[rn][cn] * d[rn][cn]
print(e)