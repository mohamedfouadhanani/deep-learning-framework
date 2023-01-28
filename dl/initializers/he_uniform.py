import numpy as np
from dl.initializers.initializer import Initializer

class HeUniform(Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape):
        n_inputs, _ = shape
        bound = np.sqrt(6 / n_inputs)
        
        return np.random.uniform(-bound, bound, shape)