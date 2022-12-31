import numpy as np

from dl.functions.loss.loss_function import LossFunction
from dl.automatic_gradient.functions.logarithm import Log

class CategoricalCrossEntropy(LossFunction):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer
        
    def __call__(self, y_pred, y):
        log_yp = Log.run(y_pred)
        yt_log_yp = y * log_yp
        loss = -1 * np.sum(yt_log_yp) / len(y)

        return loss