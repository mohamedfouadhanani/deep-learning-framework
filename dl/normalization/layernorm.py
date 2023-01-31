import numpy as np

from dl.layer import Layer

class LayerNormalization(Layer):
    EPSILON = 1e-5

    def __init__(self, is_trainable=True):
        cache = {}
        cache["gamma"] = np.random.normal(0, 1)
        cache["beta"] = np.random.normal(0, 1)

        params = ["gamma", "beta"]
        
        super().__init__(is_trainable, cache, params)
    
    def forward(self, inputs, is_optimizing):
        self.cache["inputs"] = inputs

        self.cache["mean"] = np.mean(self.cache["inputs"], axis=-1, keepdims=True)
        self.cache["sigma"] = np.std(self.cache["inputs"], axis=-1, keepdims=True)

        self.cache["inputs_normalized"] = (self.cache["inputs"] - self.cache["mean"]) / (self.cache["sigma"] + LayerNormalization.EPSILON)
        self.cache["outputs"] = self.cache["gamma"] * self.cache["inputs_normalized"] + self.cache["beta"]
        return self.cache["outputs"]

    def backward(self, doutputs):
        N, D = self.cache["inputs"].shape

        self.cache["dbeta"] = np.sum(doutputs, axis=0)
        self.cache["dgamma"] = np.sum(doutputs * self.cache["inputs_normalized"], axis=0)

        self.cache["dinputs"] = (1 / (N * self.cache["sigma"] + LayerNormalization.EPSILON)) * (N * doutputs - np.sum(doutputs, axis=0) - (self.cache["inputs"] - self.cache["mean"]) * np.sum(doutputs * (self.cache["inputs"] - self.cache["mean"]), axis=0))
        return self.cache["dinputs"]