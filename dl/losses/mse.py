import numpy as np
from dl.losses.loss import Loss

class MSE(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        return (1 / 2) * np.mean((predictions - actuals) ** 2)
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return (1 / m) * (predictions - actuals)