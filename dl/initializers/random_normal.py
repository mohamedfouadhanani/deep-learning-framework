import numpy as np
from dl.initializers.initializer import Initializer

class RandomNormal(Initializer):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, shape):
        return np.random.normal(self.mean, self.std, shape)