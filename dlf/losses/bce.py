import numpy as np

from dlf.losses.loss import Loss

class BinaryCrossEntropy(Loss):
    
    def __init__(self) -> None:
        def f(predictions, actuals):
            m, _ = actuals.shape
            return -np.sum(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions)) / m
        
        def df(predictions, actuals):
            m, _ = actuals.shape
            return ((1 - actuals) / (1 - predictions) - actuals / predictions) / m
        
        super().__init__(f, df)