import numpy as np

from dlf.initializers.initializer import Initializer

class RandomUniform(Initializer):
    
    def __init__(self, minimum = 0, maximum = 1) -> None:
        super().__init__()
        
        self.minimum = minimum
        self.maximum = maximum

    def initialize(self, shape = None):
        if shape is None:
            return np.random.uniform(self.minimum, self.maximum)
            
        return np.random.uniform(self.minimum, self.maximum, shape)