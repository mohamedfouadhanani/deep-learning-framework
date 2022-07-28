import numpy as np

from dl.functions import Function


class CategoricalCrossEntropy(Function):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer

    def __call__(self, y_hat, y):
        # y_hat is values returned from a Linear activation function output layer

        # apply softmax
        y_hat_exponentiate = np.exp(y_hat - np.max(y_hat, axis=0, keepdims=True))
        normalization_value = np.sum(y_hat_exponentiate, axis=0, keepdims=True)
        y_hat_softmax = y_hat_exponentiate / normalization_value

        # clip values
        y_hat_clip = np.clip(y_hat_softmax, 1e-7, 1 - 1e-7)

        # compute loss and return it

        correct_confidences = np.sum(y_hat_clip * y, axis=0)
        log_loss = np.log(correct_confidences)

        return -np.mean(log_loss)

    def prime(self, y_hat, y):
        _, m = y.shape
        return 1 / m * (y_hat - y)
