import numpy as np

from dlf.activation_functions.activation_function import ActivationFunction

class ELU(ActivationFunction):
    def __init__(self, alpha=0.1) -> None:
        self.alpha = alpha

        def f(inputs):
            outputs = (inputs >= 0) * inputs + (inputs < 0) * self.alpha * (np.exp(inputs) - 1)
            return outputs

        def df(inputs):
            outputs = (inputs >= 0) * 1 + (inputs < 0) * self.alpha * np.exp(inputs)
            return outputs
        
        super().__init__(f, df)