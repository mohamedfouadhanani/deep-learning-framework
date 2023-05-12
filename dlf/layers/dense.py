import numpy as np

from dlf.layers.trainable_layer import TrainableLayer
from dlf.initializers.initializer import Initializer

class Dense(TrainableLayer):
    
    def __init__(self, n_inputs, n_units, initializer: Initializer = None) -> None:
        params = ["W", "b"]
        
        super().__init__(params, initializer)
        
        self.n_units = n_units
        self.n_inputs = n_inputs
        
        W_shape = (self.n_inputs, self.n_units)
        self.cache["W"] = self.initializer.initialize(W_shape)

        b_shape = (1, self.n_units)
        self.cache["b"] = self.initializer.initialize(b_shape)
        
    def forward(self, inputs, is_optimizing=True):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.cache["W"]) + self.cache["b"]
        return self.outputs
    
    def backward(self, doutputs):
        self.doutputs = doutputs
        
        self.cache["db"] = doutputs.sum(axis=0)
        self.cache["dW"] = np.dot(self.inputs.T, doutputs)
        
        self.dinputs = np.dot(doutputs, self.cache["W"].T)
        
        return self.dinputs