import numpy as np


class L1:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, m, W):
        return self._lambda / m * np.sum(np.absolute(W))

    def prime(self, m, W):
        return self._lambda / m * np.sign(W)
