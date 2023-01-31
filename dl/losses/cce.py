import numpy as np
from dl.losses.loss import Loss

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        pass
    
    def forward(self, predictions, actuals):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        log_loss = np.log(np.sum(predictions * actuals, axis=1))

        return -np.mean(log_loss)
    
    def backward(self, predictions, actuals):
        m, _ = actuals.shape
        return - (1 / m) * (actuals / predictions)