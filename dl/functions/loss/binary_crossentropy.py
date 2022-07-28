import numpy as np

from dl.functions import Function


class BinaryCrossEntropy(Function):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer

    def __call__(self, y_hat, y):
        y_hat_clip = np.clip(y_hat, 1e-7, 1-1e-7)
        log_probs = y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip)
        return -np.mean(log_probs)

    def prime(self, y_hat, y):
        _, m = y.shape
        y_hat_clip = np.clip(y_hat, 1e-7, 1-1e-7)
        return 1 / m * (y_hat_clip - y) / (y_hat_clip * (1 - y_hat_clip))
