import numpy as np
from dl.layer import Layer

class Dropout(Layer):
    def __init__(self, keep_prob):
        cache = {}
        cache["D"] = None
        cache["keep_prob"] = keep_prob
        
        super().__init__(False, cache)
    
    def forward(self, inputs, is_optimizing):
        outputs = inputs.copy()

        if not is_optimizing:
            return outputs

        self.cache["D"] = np.random.uniform(size=inputs.shape)
        self.cache["D"] = (self.cache["D"] <= self.cache["keep_prob"])
        outputs = self.cache["D"] * inputs / self.cache["keep_prob"]
        return outputs
    
    def backward(self, doutputs):
        self.dinputs = doutputs * self.cache["D"] / self.cache["keep_prob"]
        return self.dinputs