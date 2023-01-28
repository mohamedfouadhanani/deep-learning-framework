import numpy as np
from dl.initializers.initializer import Initializer

class HeNormal(Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape):
        mean = 0

        n_inputs, _ = shape
        std = np.sqrt(2 / n_inputs)
        
        return np.random.normal(mean, std, shape)