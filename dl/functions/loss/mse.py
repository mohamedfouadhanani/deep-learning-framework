import numpy as np

from dl.functions import Function


class MSE(Function):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer

    def __call__(self, y_hat, y):
        _, m = y.shape
        loss = (1 / 2) * np.mean((y_hat - y) ** 2)

        if self.regularizer is not None:
            for l in range(1, self.model.L):
                loss += self.model.loss.regularizer(m, self.model.layers[l].W)

        return loss

    def prime(self, y_hat, y):
        _, m = y.shape
        return 1 / m * (y_hat - y)
