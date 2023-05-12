import numpy as np

from dlf.initializers.initializer import Initializer

class RandomNormal(Initializer):
    
    def __init__(self, mean = 0, std = 1) -> None:
        super().__init__()
        
        self.mean = mean
        self.std = std

    def initialize(self, shape = None):
        if shape is None:
            return np.random.normal(self.mean, self.std)

        return np.random.normal(self.mean, self.std, shape)