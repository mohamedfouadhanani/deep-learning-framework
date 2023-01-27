import numpy as np
from dl.losses.loss import Loss

class MAE(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        return np.mean(np.absolute(predictions - actuals))
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return np.sign(predictions - actuals) / m