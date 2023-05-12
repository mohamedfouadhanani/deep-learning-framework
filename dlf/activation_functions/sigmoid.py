import numpy as np
from dlf.activation_functions.activation_function import ActivationFunction

class Sigmoid(ActivationFunction):
    def __init__(self):
        def f(inputs):
            outputs = 1 / (1 + np.exp(-inputs))
            return outputs

        def df(inputs):
            activations = f(inputs)
            outputs = activations * (1 - activations)
            return outputs
        
        super().__init__(f, df)