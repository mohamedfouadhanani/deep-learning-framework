import numpy as np

from dlf.initializers.initializer import Initializer

class XavierUniform(Initializer):
    
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, shape = None):
        n_inputs, n_outputs = shape
        bound = np.sqrt(6 / (n_inputs + n_outputs))

        if shape is None:
            return np.random.uniform(-bound, bound)
        return np.random.uniform(-bound, bound, shape)