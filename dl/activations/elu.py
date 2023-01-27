import numpy as np
from dl.activations.activation import Activation

class ELU(Activation):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

        def f(inputs):
            outputs = (inputs >= 0) * inputs + (inputs < 0) * self.alpha * (np.exp(inputs) - 1)
            return outputs

        def df(inputs):
            outputs = (inputs >= 0) * 1 + (inputs < 0) * self.alpha * np.exp(inputs)
            return outputs
        
        super().__init__(f, df)