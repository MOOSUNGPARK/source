from dataset.mnist import load_mnist
import numpy as np

def mean_squared_error(y,t):
    if len(y.shape) == 1 :
        return 0.5 * np.sum((y-t)**2)
    else :
        return 0.5 * np.sum((y-t)**2, axis=1)

def cross_entropy_error(y,t):
    delta = 1e-7
    if len(y.shape) == 1 :
        return -np.sum(t * np.log(y+delta))
    else :
        return -np.sum(t * np.log(y+delta), axis=1)

t = np.array([0,0,1,0,0,0,0,0,0,0])    # 숫자2
y1 = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
y2 = np.array([0.1,0.05,0.2,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
y3 = np.array([0.0,0.05,0.3,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
y4 = np.array([0.0,0.05,0.4,0.0,0.05,0.0,0.0,0.5,0.0,0.0])
y5 = np.array([0.0,0.05,0.5,0.0,0.05,0.0,0.0,0.4,0.0,0.0])
y6 = np.array([0.0,0.05,0.6,0.0,0.05,0.0,0.0,0.3,0.0,0.0])
y7 = np.array([0.0,0.05,0.7,0.0,0.05,0.0,0.0,0.2,0.0,0.0])
y8 = np.array([0.0,0.1,0.8,0.0,0.1,0.0,0.0,0.2,0.0,0.0])
y9 = np.array([0.0,0.05,0.9,0.0,0.05,0.0,0.0,0.0,0.0,0.0])

all_y = np.array([eval('y'+str(i)) for i in range(1,10)])

print(cross_entropy_error(all_y,t))

print(np.random.choice(60000,10))
print(np.random.rand(10))
print(np.random.randn(10))



(x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True, flatten=True)
train_size = 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(x_train[batch_mask])
print(t_train[batch_mask])