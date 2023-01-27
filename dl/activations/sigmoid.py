import numpy as np
from dl.activations.activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def f(inputs):
            outputs = 1 / (1 + np.exp(-inputs))
            return outputs

        def df(inputs):
            activations = f(inputs)
            outputs = activations * (1 - activations)
            return outputs
        
        super().__init__(f, df)