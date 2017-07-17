import numpy as np
import matplotlib.pyplot as plt

chi = np.loadtxt('c:\\python\\data\\창업건수.csv', skiprows=1, unpack=True, delimiter=',')
print(chi)
x = chi[0]
y = chi[4]
plt.figure()
plt.plot(x,y)
plt.xlabel('year')
plt.ylabel('chicken')
plt.title('year & chicken')
plt.show()


