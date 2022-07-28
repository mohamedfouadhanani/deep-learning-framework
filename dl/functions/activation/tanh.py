import numpy as np

from dl.functions import Function


class TanH(Function):
    def __call__(self, Z):
        return np.tanh(Z)

    def prime(self, Z):
        A = np.tanh(Z)
        return 1 - A ** 2
