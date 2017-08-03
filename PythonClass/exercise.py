import numpy as np

def padding(input, inputsize, outputsize, filtersize, stride=1):
    padsize = []
    for idx in range(len(inputsize)):
        padsize.append((stride * (outputsize[idx] - 1) - inputsize[idx] + filtersize[idx])/2)
    return np.pad(input,
                  pad_width=((round(padsize[0]+1e-7), int(padsize[0])),
                             (round(padsize[1]+1e-7), int(padsize[1]))),
                  mode='constant')

a = np.arange(120).reshape(15,8)
a_pad = np.array([padding(a, (15,8), (15,8), (3,3)) for i in range(10)])
f = np.array([np.arange(9) for j in range(10)]).reshape(10,3,3)
result = np.zeros(120)

for fn in range(a_pad.shape[0]):
    temp=[]
    for rn in range(a_pad.shape[1]- (f.shape[1]-1)):
        for cn in range(a_pad.shape[2]- (f.shape[2]-1)):
            temp.append(np.sum(a_pad[fn, rn : rn+f.shape[1], cn : cn+f.shape[2]] * f[fn]))
    result += temp

print(np.array(result).reshape(15,8))