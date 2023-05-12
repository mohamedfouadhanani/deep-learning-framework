import numpy as np

from dlf.losses.loss import Loss

class CategoricalCrossEntropy(Loss):
    def __init__(self) -> None:
        LIMIT = 1e-15

        def f(predictions, actuals):
            exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            softmax = exps / exps.sum(axis=1, keepdims=True)
            clipped_predictions = np.clip(softmax, LIMIT, 1 - LIMIT)
            
            return -np.mean(np.sum(actuals * np.log(clipped_predictions), axis=1))
            
        def df(predictions, actuals):
            m, _ = actuals.shape
            return (1 / m) * (predictions - actuals)
        
        super().__init__(f, df)