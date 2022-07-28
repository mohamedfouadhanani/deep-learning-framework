import numpy as np

from dl.functions import Function


class ReLU(Function):
    def __call__(self, Z):
        return np.maximum(0, Z)

    def prime(self, Z):
        return (Z > 0).astype(int)
