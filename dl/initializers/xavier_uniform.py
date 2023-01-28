import numpy as np
from dl.initializers.initializer import Initializer

class XavierUniform(Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape):
        n_inputs, n_outputs = shape
        bound = np.sqrt(6 / (n_inputs + n_outputs))

        return np.random.uniform(-bound, bound, shape)