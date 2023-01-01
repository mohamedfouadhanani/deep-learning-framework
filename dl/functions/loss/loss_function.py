from dl.regularization.regularizer import Regularizer
from dl.automatic_gradient.variable import Variable

class LossFunction:
    def __init__(self, regularizer: Regularizer):
        pass
        
    def __call__(self, y_pred, y) -> Variable:
        pass
    
    def __repr__(self):
        pass