import numpy as np
from dl.losses.loss import Loss

class BinaryCrossEntropy(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        m, _ = actuals.shape
        return -np.sum(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions)) / m
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return ((1 - actuals) / (1 - predictions) - actuals / predictions) / m