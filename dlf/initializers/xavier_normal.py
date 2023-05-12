import numpy as np

from dlf.initializers.initializer import Initializer

class XavierNormal(Initializer):
    
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, shape = None):
        mean = 0

        n_inputs, n_outputs = shape
        std = np.sqrt(2 / (n_inputs + n_outputs))
        
        if shape is None:
            return np.random.normal(mean, std)
            
        return np.random.normal(mean, std, shape)