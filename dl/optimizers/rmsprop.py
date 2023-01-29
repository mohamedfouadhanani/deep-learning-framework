import numpy as np

from dl.optimizers.optimizer import Optimizer

class RMSProp(Optimizer):
    EPSILON = 1e-7

    def __init__(self, learning_rate, batch_size, beta, lr_decay=lambda lr0: lr0):
        super().__init__(learning_rate, batch_size, lr_decay)
        self.beta = beta
    
    def __call__(self, layer):
        if not np.all([f"sd{param}" in layer.cache for param in layer.params]):
            for param in layer.params:
                layer.cache[f"sd{param}"] = np.zeros_like(layer.cache[param])
        
        for param in layer.params:
            layer.cache[f"sd{param}"] = self.beta * layer.cache[f"sd{param}"] + (1 - self.beta) * layer.cache[f"d{param}"] ** 2
            layer.cache[param] -= self.learning_rate * layer.cache[f"d{param}"] / np.sqrt(layer.cache[f"sd{param}"] + RMSProp.EPSILON)

        self.learning_rate = self.lr_decay(self.learning_rate0)