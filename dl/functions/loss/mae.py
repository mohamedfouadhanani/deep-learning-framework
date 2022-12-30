from dl.functions.loss.loss_function import LossFunction

class MSE(LossFunction):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer
        
    def __call__(self, y_pred, y):
        pass