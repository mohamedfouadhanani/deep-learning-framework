import numpy as np

from dlf.layers.trainable_layer import TrainableLayer
from dlf.initializers.random_normal import RandomNormal

class BatchNormalization(TrainableLayer):
    EPSILON = 1e-5
    
    def __init__(self) -> None:
        params = ["gamma", "beta"]
        super().__init__(params, RandomNormal())
        
        self.cache["gamma"] = self.initializer.initialize()
        self.cache["beta"] = self.initializer.initialize()
    
    def forward(self, inputs, is_optimizing=True):
        self.inputs = inputs

        self.cache["mean"] = np.mean(self.inputs, axis=0)
        self.cache["variance"] = np.var(self.inputs, axis=0)

        self.cache["inputs_normalized"] = (self.inputs - self.cache["mean"]) / np.sqrt(self.cache["variance"] + BatchNormalization.EPSILON)

        self.outputs = self.cache["gamma"] * self.cache["inputs_normalized"] + self.cache["beta"]
        return self.outputs
    
    def backward(self, doutputs):
        self.doutputs = doutputs
        
        N, _ = self.inputs.shape

        dinputs_normalized = doutputs * self.cache["gamma"]

        dvariance = np.sum(dinputs_normalized * (self.inputs - self.cache["mean"]) * -0.5 * (self.cache["variance"] + BatchNormalization.EPSILON)**-1.5, axis=0)

        dmean = np.sum(dinputs_normalized * -1 / np.sqrt(self.cache["variance"] + BatchNormalization.EPSILON), axis=0) + dvariance * np.mean(-2 * (self.inputs - self.cache["mean"]), axis=0)
        
        self.dinputs = dinputs_normalized / np.sqrt(self.cache["variance"] + BatchNormalization.EPSILON) + dvariance * 2 * (self.inputs - self.cache["mean"]) / N + dmean / N

        self.cache["dgamma"] = np.sum(doutputs * self.cache["inputs_normalized"], axis=0)
        self.cache["dbeta"] = np.sum(doutputs, axis=0)

        return self.dinputs