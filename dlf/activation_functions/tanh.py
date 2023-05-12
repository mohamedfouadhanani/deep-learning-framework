import numpy as np
from dlf.activation_functions.activation_function import ActivationFunction

class TanH(ActivationFunction):
    def __init__(self):
        def f(inputs):
            outputs = np.tanh(inputs)
            return outputs

        def df(inputs):
            activations = f(inputs)
            outputs = 1 - activations ** 2
            return outputs
        
        super().__init__(f, df)