from dl.functions.loss.loss_function import LossFunction

class MSE(LossFunction):
    def __init__(self, regularizer=None):
        self.regularizer = regularizer
        self.model = None
        
    def __call__(self, y_pred, y):
        loss = ((y_pred - y) ** 2).sum() / (2 * len(y))

        if self.regularizer is not None:
            regularization = self.regularizer(m=len(y), params=self.model.parameters())
            
            total_loss = loss + regularization
            
            return total_loss
        
        return loss
    
    def __repr__(self):
        return f"MSE(regularizer={self.regularizer})"