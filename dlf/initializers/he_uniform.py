import numpy as np

from dlf.initializers.initializer import Initializer

class HeUniform(Initializer):
    
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, shape = None):
        n_inputs, _ = shape
        bound = np.sqrt(6 / n_inputs)
        
        if shape is None:
            return np.random.uniform(-bound, bound, shape)
        
        return np.random.uniform(-bound, bound, shape)