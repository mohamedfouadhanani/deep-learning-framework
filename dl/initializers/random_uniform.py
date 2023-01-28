import numpy as np
from dl.initializers.initializer import Initializer

class RandomUniform(Initializer):
    def __init__(self, minimum=0, maximum=1):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
    
    def __call__(self, shape):
        return np.random.uniform(self.minimum, self.maximum, shape)