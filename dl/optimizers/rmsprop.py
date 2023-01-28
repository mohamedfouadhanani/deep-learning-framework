import numpy as np

from dl.optimizers.optimizer import Optimizer

class RMSProp(Optimizer):
    EPSILON = 1e-7

    def __init__(self, learning_rate, batch_size, beta, lr_decay=lambda lr0: lr0):
        super().__init__(learning_rate, batch_size, lr_decay)
        self.beta = beta
    
    def __call__(self, layer):
        if not hasattr(layer, "vdW") and not hasattr(layer, "vdb"):
            layer.sdW = np.zeros_like(layer.W)
            layer.sdb = np.zeros_like(layer.b)
        
        layer.sdW = self.beta * layer.sdW + (1 - self.beta) * layer.dW ** 2
        layer.sdb = self.beta * layer.sdb + (1 - self.beta) * layer.db ** 2

        layer.W -= self.learning_rate * layer.dW / np.sqrt(layer.sdW + RMSProp.EPSILON)
        layer.b -= self.learning_rate * layer.db / np.sqrt(layer.sdb + RMSProp.EPSILON)

        self.learning_rate = self.lr_decay(self.learning_rate0)