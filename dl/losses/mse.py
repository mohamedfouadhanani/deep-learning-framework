import numpy as np
from dl.losses.loss import Loss

class MSE(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        m, _ = actuals.shape
        return (1 / 2) * np.sum((predictions - actuals) ** 2) / m
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return (1 / m) * (predictions - actuals)