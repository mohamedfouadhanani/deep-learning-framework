import numpy as np
from dl.losses.loss import Loss

class MAE(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        m, _ = actuals.shape
        return np.sum(np.absolute(predictions - actuals)) / m
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return np.sign(predictions - actuals) / m