from common.functions import mean_squared_error_axis1
import numpy as np
a = np.array([[100], [200]])
b = np.array([[105], [200]])
c = np.array([[100], [200]])

print(mean_squared_error_axis1(a, b))
print(mean_squared_error_axis1(a, c))