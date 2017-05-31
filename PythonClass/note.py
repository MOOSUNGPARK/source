import numpy as np

a = [[12,3,4,5,6],[12,3,4,5,0],[12,3,4,5,6],[12,3,4,5,6],[12,3,4,-1,100]]
ab= np.array(a)
abc = []
for rownum in range(len(a)):
    for colnum in range(len(a[0])):
        if a[rownum][colnum] == np.max(ab) :
            abc.append([rownum,colnum])

print(abc)

print(np.sum(ab))

print(np.count_nonzero(ab[:][:2]))

