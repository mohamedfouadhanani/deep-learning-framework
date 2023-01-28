import numpy as np

from dl.optimizers.optimizer import Optimizer

class MomentumGradientDescent(Optimizer):
    def __init__(self, learning_rate, batch_size, beta, lr_decay=lambda lr0: lr0):
        super().__init__(learning_rate, batch_size, lr_decay)
        self.beta = beta
    
    def __call__(self, layer):
        if not hasattr(layer, "vdW") and not hasattr(layer, "vdb"):
            layer.vdW = np.zeros_like(layer.W)
            layer.vdb = np.zeros_like(layer.b)
        
        layer.vdW = self.beta * layer.vdW + (1 - self.beta) * layer.dW
        layer.vdb = self.beta * layer.vdb + (1 - self.beta) * layer.db

        layer.W -= self.learning_rate * layer.dW
        layer.b -= self.learning_rate * layer.db

        self.learning_rate = self.lr_decay(self.learning_rate0)