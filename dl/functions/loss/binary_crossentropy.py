import numpy as np

from dl.automatic_gradient.variable import Variable
from dl.regularization.regularizer import Regularizer
from dl.functions.loss.loss_function import LossFunction
from dl.automatic_gradient.functions.logarithm import Log

class BinaryCrossEntropy(LossFunction):
    def __init__(self, regularizer: Regularizer=None):
        self.regularizer = regularizer
        self.model = None
        
    def __call__(self, y_pred, y) -> Variable:
        log_y_pred = Log.run(y_pred)
        inverted_y = (1 - y)
        inverted_yp = (1 - y_pred)
        log_inverted_yp = Log.run(inverted_yp)
        y_log_yp = y * log_y_pred
        inverted_y_log_inverted_yp = inverted_y * log_inverted_yp
        
        loss: Variable = -1 * np.sum(y_log_yp + inverted_y_log_inverted_yp) / len(y)

        if self.regularizer is not None:
            regularization = self.regularizer(m=len(y), params=self.model.parameters(including_biases=False))
            
            total_loss: Variable = loss + regularization
            
            return total_loss
        
        return loss
    
    def __repr__(self) -> str:
        return f"BinaryCrossEntropy(regularizer={self.regularizer})"