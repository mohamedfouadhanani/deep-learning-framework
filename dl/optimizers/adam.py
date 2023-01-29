import numpy as np

from dl.optimizers.optimizer import Optimizer

class AdaptiveMomentEstimation(Optimizer):
    EPSILON = 1e-7

    def __init__(self, learning_rate, batch_size, beta1, beta2, lr_decay=lambda lr0: lr0):
        super().__init__(learning_rate, batch_size, lr_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.coefficient = 0
    
    def __call__(self, layer):
        if not np.all([f"vd{param}" in layer.cache for param in layer.params]) or not np.all([f"sd{param}" in layer.cache for param in layer.params]):
            for param in layer.params:
                layer.cache[f"vd{param}"] = np.zeros_like(layer.cache[param])
                layer.cache[f"vd{param}_corrected"] = None

                layer.cache[f"sd{param}"] = np.zeros_like(layer.cache[param])
                layer.cache[f"sd{param}_corrected"] = None
        
        for param in layer.params:
            layer.cache[f"vd{param}"] = self.beta1 * layer.cache[f"vd{param}"] + (1 - self.beta1) * layer.cache[f"d{param}"]
            layer.cache[f"vd{param}_corrected"] = layer.cache[f"vd{param}"] / (1 - self.beta1 ** (self.coefficient + 1))
            
            layer.cache[f"sd{param}"] = self.beta2 * layer.cache[f"sd{param}"] + (1 - self.beta2) * layer.cache[f"d{param}"] ** 2
            layer.cache[f"sd{param}_corrected"] = layer.cache[f"sd{param}"] / (1 - self.beta2 ** (self.coefficient + 1))

            layer.cache[param] -= self.learning_rate * layer.cache[f"vd{param}_corrected"] / np.sqrt(layer.cache[f"sd{param}_corrected"] + AdaptiveMomentEstimation.EPSILON)

        self.coefficient += 1
        self.learning_rate = self.lr_decay(self.learning_rate0)