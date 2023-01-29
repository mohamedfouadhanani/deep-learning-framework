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

        if is_optimizing:
            self.cache["batch_mean"] = np.mean(self.cache["inputs"], axis=0)
            self.cache["batch_variance"] = np.var(self.cache["inputs"], axis=0)

            self.cache["inputs_normalized"] = (self.cache["inputs"] - self.cache["batch_mean"]) / np.sqrt(self.cache["batch_variance"] + BatchNormalization.EPSILON)

            if "running_mean" not in self.cache or "running_variance" not in self.cache:
                self.cache["running_mean"] = np.zeros_like(self.cache["batch_mean"])
                self.cache["running_variance"] = np.zeros_like(self.cache["batch_variance"])

            self.cache["running_mean"] = self.cache["momentum"] * self.cache["running_mean"] + (1 - self.cache["momentum"]) * self.cache["batch_mean"]
            self.cache["running_variance"] = self.cache["momentum"] * self.cache["running_variance"] + (1 - self.cache["momentum"]) * self.cache["batch_variance"]
        else:
            self.cache["inputs_normalized"] = (self.cache["inputs"] - self.cache["running_mean"]) / np.sqrt(self.cache["running_variance"] + BatchNormalization.EPSILON)
        
        outputs = self.cache["gamma"] * self.cache["inputs_normalized"] + self.cache["beta"]
        return outputs

    def backward(self, doutputs):
        self.cache["dgamma"] = np.sum(doutputs * self.cache["inputs_normalized"], axis=0)
        self.cache["dbeta"] = np.sum(doutputs, axis=0)

        m, _ = self.cache["inputs"].shape
        t = 1 / np.sqrt(self.cache["batch_variance"] + BatchNormalization.EPSILON)

        self.cache["dinputs"] = (self.cache["gamma"] * t / m) * (m * doutputs - np.sum(doutputs, axis=0) - t**2 * (self.cache["inputs"] - self.cache["batch_mean"]) * np.sum(doutputs * (self.cache["inputs"] - self.cache["batch_mean"]), axis=0))
        return self.cache["dinputs"]