from dl.functions.loss.loss_function import LossFunction
from dl.automatic_gradient.functions.absolute import Absolute

class MAE(LossFunction):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer
        self.model = None
        
    def __call__(self, y_pred, y):
        loss = Absolute.run(y_pred - y).sum() / len(y)

        if self.regularizer is not None:
            regularization = self.regularizer(m=len(y), params=self.model.parameters())
            
            total_loss = loss + regularization
            
            return total_loss
        
        return loss
    
    def __repr__(self):
        return f"MAE(regularizer={self.regularizer})"