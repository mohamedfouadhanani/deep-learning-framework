from dl.functions.loss.loss_function import LossFunction
from dl.automatic_gradient.functions.absolute import Absolute

class MSE(LossFunction):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer
        
    def __call__(self, y_pred, y):
        return Absolute.run(y_pred - y).sum() / len(y)