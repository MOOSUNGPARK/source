import numpy as np

# read data
##data = pd.read_csv('ex1data2.txt', header=None)
data = np.loadtxt('c:\python\data\ex1data2.txt', delimiter=',')
x = data[:, 0:2]
y = data[:, 2]
m = len(y)
y = y.reshape((m, 1))



# normalize the features
def normalize(x):
    n = len(x[0])
    for i in range(n):
        mean = np.mean(x[:, i])
        std = np.std(x[:, i])
        x[:, i] = (x[:, i] - mean) / std
    return x


x = normalize(x)

# add intercept term to X
X = np.ones((m, 3))
X[:, 1:] = x  # 47*3
print(X)
# Loss function
##initialize the parameters
theta = np.zeros((3, 1))  # 3*1
print(theta)
num_iters = 20000
alpha = 0.01


def loss(x, y, theta):
    L = np.sum((x.dot(theta) - y) ** 2) / (2 * m)
    return L


loss(X, y, theta)


# create a gradient descent function
def gradient_descent(x, y, theta, alpha, num_iters):
    p = len(x[0])
    L_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        for f in range(p):
            dtheta = np.sum((x.dot(theta) - y) * x[:, f]) / m

            theta[f][0] = theta[f][0] - alpha * dtheta

    L_history[i, 0] = loss(x, y, theta)

    return theta, L_history


gradient_descent(X, y, theta, alpha, num_iters)

a = np.array((1, 1650, 3))
predict = a.dot(theta)
print(predict)





---------------------------------------
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

data=loadtxt('ex1data1.txt',delimiter=",")

scatter(data[:,0],data[:,1],marker='o',c='b')
title("profits distribution")
xlabel("Population of City in 10,000s")
ylabel('Profit in $10,0000s')
show()

X=data[:,0]
y=data[:,1]

m=y.size

it=ones(shape=(m,2))
it[:,1]=X

theta=zeros(shape=(2,1))

iterations=1500
alpha=0.01

def compute_cost(X,y,theta):
	m=y.size
	predictions=X.dot(theta).flatten()

	sqErrors=(predictions-y)**2
	J=(1.0/(2*m))*sqErrors.sum()
	return J
def gradient_descent(X,y,theta,alpha,num_iters):
	m=y.size
	J_history=zeros(shape=(num_iters,1))

	for i in range(num_iters):
		predictions=X.dot(theta).flatten()

		errors_x1=(predictions-y)*X[:,0]
		errors_x2=(predictions-y)*X[:,1]

		theta[0][0]=theta[0][0]-alpha*(1.0/m)*errors_x1.sum()
		theta[1][0]=theta[1][0]-alpha*(1.0/m)*errors_x2.sum()

		J_history[i,0]=compute_cost(X,y,theta)
	return theta, J_history



