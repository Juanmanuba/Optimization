import numpy as np
x = [[1, 2, 3, 4, 1, 32, 3, 3, 4, 35]]

index1 = np.argwhere(np.ravel(x) < 4)

print(np.ravel(index1))
