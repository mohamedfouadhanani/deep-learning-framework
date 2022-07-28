import numpy as np

from dl.functions import Function


class LeakyReLU(Function):
    a = 0.01

    def __call__(self, Z):
        return np.maximum(LeakyReLU.a * Z, Z)

    def prime(self, Z):
        return np.where(Z > 0, 1, LeakyReLU.a)
