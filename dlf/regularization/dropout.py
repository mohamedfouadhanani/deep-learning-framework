import numpy as np

from dlf.layers.layer import Layer

class Dropout(Layer):
    def __init__(self, keep_prob) -> None:
        super().__init__()
        self.keep_prob = keep_prob
    
    def forward(self, inputs, is_optimizing=True):
        self.inputs = inputs
        
        self.outputs = inputs.copy()

        if not is_optimizing:
            return self.outputs

        self.cache["D"] = np.random.uniform(size=inputs.shape)
        self.cache["D"] = (self.cache["D"] <= self.keep_prob)
        
        self.outputs = self.cache["D"] * inputs / self.keep_prob
        
        return self.outputs
    
    def backward(self, doutputs):
        self.doutputs = doutputs
        self.dinputs = doutputs * self.cache["D"] / self.keep_prob
        return self.dinputs