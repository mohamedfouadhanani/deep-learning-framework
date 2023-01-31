import numpy as np

from dl.layer import Layer

class BatchNormalization(Layer):
    EPSILON = 1e-5

    def __init__(self, momentum=0.9, is_trainable=True):
        cache = {}
        cache["gamma"] = np.random.normal(0, 1)
        cache["beta"] = np.random.normal(0, 1)
        cache["momentum"] = momentum

        params = ["gamma", "beta"]
        
        super().__init__(is_trainable, cache, params)
    
    def forward(self, inputs, is_optimizing):
        self.cache["inputs"] = inputs

        self.cache["mean"] = np.mean(self.cache["inputs"], axis=0)
        self.cache["variance"] = np.var(self.cache["inputs"], axis=0)

        self.cache["inputs_normalized"] = (self.cache["inputs"] - self.cache["mean"]) / np.sqrt(self.cache["variance"] + BatchNormalization.EPSILON)

        self.cache["outputs"] = self.cache["gamma"] * self.cache["inputs_normalized"] + self.cache["beta"]
        return self.cache["outputs"]

    def backward(self, doutputs):
        N, D = self.cache["inputs"].shape

        dinputs_normalized = doutputs * self.cache["gamma"]

        dvariance = np.sum(dinputs_normalized * (self.cache["inputs"] - self.cache["mean"]) * -0.5 * (self.cache["variance"] + BatchNormalization.EPSILON)**-1.5, axis=0)

        dmean = np.sum(dinputs_normalized * -1 / np.sqrt(self.cache["variance"] + BatchNormalization.EPSILON), axis=0) + dvariance * np.mean(-2 * (self.cache["inputs"] - self.cache["mean"]), axis=0)
        
        self.cache["dinputs"] = dinputs_normalized / np.sqrt(self.cache["variance"] + BatchNormalization.EPSILON) + dvariance * 2 * (self.cache["inputs"] - self.cache["mean"]) / N + dmean / N

        self.cache["dgamma"] = np.sum(doutputs * self.cache["inputs_normalized"], axis=0)
        self.cache["dbeta"] = np.sum(doutputs, axis=0)

        return self.cache["dinputs"]