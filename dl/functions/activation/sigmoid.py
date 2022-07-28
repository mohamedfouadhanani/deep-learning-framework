import numpy as np

from dl.functions import Function


class Sigmoid(Function):
    def __call__(self, Z):
        return 1 / (1 + np.exp(-Z))

    def prime(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A * (1 - A)
