import numpy as np

from dlf.losses.loss import Loss

class MSE(Loss):
    def __init__(self) -> None:
        def f(predictions, actuals):
            m, _ = actuals.shape
            return (1 / 2) * np.sum((predictions - actuals) ** 2) / m
        
        def df(predictions, actuals):
            m, _ = actuals.shape
            return (1 / m) * (predictions - actuals)
        
        super().__init__(f, df)