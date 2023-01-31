import numpy as np

from dl.optimizers.optimizer import Optimizer

class MomentumGradientDescent(Optimizer):
    def __init__(self, learning_rate, beta, lr_decay=lambda lr0: lr0):
        super().__init__(learning_rate, lr_decay)
        self.beta = beta
    
    def __call__(self, layer):
        if not np.all([f"vd{param}" in layer.cache for param in layer.params]):
            for param in layer.params:
                layer.cache[f"vd{param}"] = np.zeros_like(layer.cache[param])
        
        for param in layer.params:
            layer.cache[f"vd{param}"] = self.beta * layer.cache[f"vd{param}"] + (1 - self.beta) * layer.cache[f"d{param}"]
            layer.cache[param] -= self.learning_rate * layer.cache[f"vd{param}"]

        self.learning_rate = self.lr_decay(self.learning_rate0)