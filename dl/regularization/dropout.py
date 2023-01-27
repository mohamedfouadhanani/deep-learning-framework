import numpy as np
from dl.layer import Layer

class Dropout(Layer):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    
    def forward(self, inputs, is_optimizing):
        outputs = inputs.copy()

        if not is_optimizing:
            return outputs

        self.D = np.random.uniform(size=inputs.shape)
        self.D = (self.D <= self.keep_prob)
        outputs = self.D * inputs / self.keep_prob
        return outputs
    
    def backward(self, doutputs):
        self.dinputs = doutputs * self.D / self.keep_prob
        return self.dinputs