import numpy as np


def objectiveFunction(x):
    return np.sum(np.square(x))


print(objectiveFunction([1, 2]))
