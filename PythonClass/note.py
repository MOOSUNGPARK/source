import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 12, 0.1)
print(t)

plt.figure()
plt.plot(t)
plt.grid()
plt.xlabel('size')
plt.ylabel('cost')
plt.title('size & cost')
plt.show()
