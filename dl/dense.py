import numpy as np

from dl.layer import Layer
from dl.initializers.random_uniform import RandomUniform

class Dense(Layer):
    def __init__(self, n_inputs, n_outputs, weights_initializer=RandomUniform(), is_trainable=True):
        cache = {}
        cache["W"] = weights_initializer((n_inputs, n_outputs))
        cache["b"] = np.random.randn(1, n_outputs)
        cache["inputs"] = None
        cache["outputs"] = None

        params = ["W", "b"]
        
        super().__init__(is_trainable, cache, params)
    
    def forward(self, inputs, is_optimizing):
        self.cache["inputs"] = inputs
        self.cache["outputs"] = np.dot(self.cache["inputs"], self.cache["W"]) + self.cache["b"]
        return self.cache["outputs"]

    def backward(self, doutputs):
        self.cache["db"] = doutputs.sum(axis=0)
        self.cache["dW"] = np.dot(self.cache["inputs"].T, doutputs)
        self.cache["dinputs"] = np.dot(doutputs, self.cache["W"].T)
        return self.cache["dinputs"]