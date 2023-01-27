import numpy as np
from dl.losses.loss import Loss

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        exps = np.exp(predictions - np.amax(predictions, axis=1, keepdims=True))
        softmax = exps / exps.sum(axis=1, keepdims=True)
        predictions = np.clip(softmax, 1e-15, 1 - 1e-15)
        
        log_loss = np.log(np.sum(predictions * actuals, axis=1))

        return -np.mean(log_loss)
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return (1 / m) * (predictions - actuals)