import numpy as np
import pandas as pd

def read_data(*filename):
    data1 = np.loadtxt(filename[0], delimiter=',')
    data2 = np.loadtxt(filename[1], delimiter=',')
    data = np.append(data1, data2, axis=0)
    np.random.shuffle(data)

# a = np.loadtxt('C:\\python\\data\\fer2013\\fer2013.csv', delimiter=',', skiprows=1, usecols=(0, 1), dtype='str')
# b = np.genfromtxt('C:\\python\\data\\fer2013\\fer2013.csv', usecols=(1), delimiter=' ')
# print(a[:,1].split(' '))
# b = []
a = np.loadtxt('C:\\python\\data\\fer2013\\fer2013.csv', delimiter=' ', skiprows=1, dtype='str')
print(a)
# for rn in range(len(a)):
#     a[rn,1].split(' ')
#
# print(b)


# c = pd.read_csv('C:\\python\\data\\fer2013\\fer2013.csv', header=None, skiprows=(1), names=['Label', 'Data', 'Type'])
# print(c['Data'])

# d = pd.read_csv('C:\\python\\data\\fer2013\\fer2013.csv', header=None, usecols=(0), skiprows=(1))


