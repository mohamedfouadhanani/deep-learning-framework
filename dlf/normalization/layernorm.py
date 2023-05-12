import numpy as np

from dlf.layers.trainable_layer import TrainableLayer
from dlf.initializers.random_normal import RandomNormal

class LayerNormalization(TrainableLayer):
    EPSILON = 1e-5
    
    def __init__(self) -> None:
        params = ["gamma", "beta"]
        super().__init__(params, RandomNormal())

        self.cache["gamma"] = self.initializer.initialize()
        self.cache["beta"] = self.initializer.initialize()
    
    def forward(self, inputs, is_optimizing=True):
        self.inputs = inputs

        self.cache["mean"] = np.mean(self.inputs, axis=-1, keepdims=True)
        self.cache["sigma"] = np.std(self.inputs, axis=-1, keepdims=True)

        self.cache["inputs_normalized"] = (self.inputs - self.cache["mean"]) / (self.cache["sigma"] + LayerNormalization.EPSILON)
        self.outputs = self.cache["gamma"] * self.cache["inputs_normalized"] + self.cache["beta"]

        return self.outputs
    
    def backward(self, doutputs):
        self.doutputs = doutputs
        
        N, _ = self.inputs.shape

        self.cache["dbeta"] = np.sum(doutputs, axis=0)
        self.cache["dgamma"] = np.sum(doutputs * self.cache["inputs_normalized"], axis=0)

        self.dinputs = (1 / (N * self.cache["sigma"] + LayerNormalization.EPSILON)) * (N * doutputs - np.sum(doutputs, axis=0) - (self.inputs - self.cache["mean"]) * np.sum(doutputs * (self.inputs - self.cache["mean"]), axis=0))
        
        return self.dinputs