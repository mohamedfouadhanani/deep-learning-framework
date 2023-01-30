import numpy as np

from dl.layer import Layer
from dl.initializers.random_uniform import RandomUniform

class Softmax(Layer):
    def __init__(self):
        cache = {}
        cache["inputs"] = None
        cache["outputs"] = None
        
        super().__init__(False, cache)
    
    def forward(self, inputs, is_optimizing):
        self.cache["inputs"] = inputs
        exps = np.exp(self.cache["inputs"] - np.max(self.cache["inputs"], axis=1, keepdims=True))
        self.cache["outputs"] = exps / np.sum(exps, axis=1, keepdims=True)
        return self.cache["outputs"]

    def backward(self, doutputs):
        self.cache["dinputs"] = np.empty_like(doutputs)
        m, _ = doutputs.shape

        for i in range(m):
            # compute jacobian
            output_i = self.cache["outputs"][i]
            output_reshaped = output_i.reshape(-1, 1)
            jacobian = np.diagflat(output_i) - np.dot(output_reshaped, output_reshaped.T)
            
            # compute dL/dx(i)
            self.cache["dinputs"][i] = np.dot(doutputs[i], jacobian)
        
        return self.cache["dinputs"]