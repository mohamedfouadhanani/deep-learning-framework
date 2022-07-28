import numpy as np


class L2:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, m, W):
        return self._lambda / (2 * m) * np.sum(W ** 2)

    def prime(self, m, W):
        return self._lambda / m * W
