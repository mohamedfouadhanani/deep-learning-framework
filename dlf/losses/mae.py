import numpy as np

from dlf.losses.loss import Loss

class MAE(Loss):
    def __init__(self) -> None:
        def f(predictions, actuals):
            m, _ = actuals.shape
            return np.sum(np.absolute(predictions - actuals)) / m
        
        def df(predictions, actuals):
            m, _ = actuals.shape
            return np.sign(predictions - actuals) / m
        
        super().__init__(f, df)