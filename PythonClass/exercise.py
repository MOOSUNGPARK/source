from numpy import zeros, dot, ones
alpha=ones(shape=(1,4))
theta=zeros(shape=(4,1))

beta = [[1],[2],[3],[4]]
print(alpha.dot(beta).flatten())