import numpy as np

from dl.layer import Layer
from dl.initializers.random_uniform import RandomUniform

class Dense(Layer):
    def __init__(self, n_inputs, n_outputs, weights_initializer=RandomUniform()):
        self.W = weights_initializer((n_inputs, n_outputs))
        self.b = np.random.randn(1, n_outputs)
    
    def forward(self, inputs, is_optimizing):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.W) + self.b
        return self.outputs

    def backward(self, doutputs):
        self.db = doutputs.sum(axis=0)
        self.dW = np.dot(self.inputs.T, doutputs)
        self.dinputs = np.dot(doutputs, self.W.T)
        return self.dinputs