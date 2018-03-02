import numpy as np

TIMES = 100000
MAX = 100
error = 0

for i in range(TIMES):
	m = np.random.rand() * MAX
	n = np.random.rand() * MAX
	error += (m - n) ** 2

error /= TIMES
print(error)

