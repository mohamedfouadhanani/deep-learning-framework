from dl.automatic_gradient.variable import Variable
from dl.regularization.regularizer import Regularizer
from dl.functions.loss.loss_function import LossFunction
from dl.automatic_gradient.functions.absolute import Absolute

class MAE(LossFunction):
    def __init__(self, regularizer: Regularizer=None):
        self.regularizer = regularizer
        self.model = None
        
    def __call__(self, y_pred, y) -> Variable:
        loss: Variable = Absolute.run(y_pred - y).sum() / len(y)

        if self.regularizer is not None:
            regularization = self.regularizer(m=len(y), params=self.model.parameters(including_biases=False))
            
            total_loss: Variable = loss + regularization
            
            return total_loss
        
        return loss
    
    def __repr__(self) -> str:
        return f"MAE(regularizer={self.regularizer})"