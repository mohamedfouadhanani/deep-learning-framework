import numpy as np

from dl.functions import Function


class Leaky_ReLU(Function):
    a = 0.01

    def __call__(self, Z):
        return np.maximum(Leaky_ReLU.a * Z, Z)

    def prime(self, Z):
        return np.where(Z > 0, 1, Leaky_ReLU.a)
