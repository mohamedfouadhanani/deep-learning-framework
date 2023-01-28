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
        if not hasattr(layer, "vdW") and not hasattr(layer, "vdb") and not hasattr(layer, "sdW") and not hasattr(layer, "sdb"):
            layer.vdW = np.zeros_like(layer.W)
            layer.vdb = np.zeros_like(layer.b)
            
            layer.sdW = np.zeros_like(layer.W)
            layer.sdb = np.zeros_like(layer.b)

            layer.vdW_corrected = None
            layer.vdb_corrected = None

            layer.sdW_corrected = None
            layer.sdb_corrected = None
        
        layer.vdW = self.beta1 * layer.vdW + (1 - self.beta1) * layer.dW
        layer.vdb = self.beta1 * layer.vdb + (1 - self.beta1) * layer.db

        layer.vdW_corrected = layer.vdW / (1 - self.beta1 ** (self.coefficient + 1))
        layer.vdb_corrected = layer.vdb / (1 - self.beta1 ** (self.coefficient + 1))
        
        layer.sdW = self.beta2 * layer.sdW + (1 - self.beta2) * layer.dW ** 2
        layer.sdb = self.beta2 * layer.sdb + (1 - self.beta2) * layer.db ** 2

        layer.sdW_corrected = layer.sdW / (1 - self.beta2 ** (self.coefficient + 1))
        layer.sdb_corrected = layer.sdb / (1 - self.beta2 ** (self.coefficient + 1))

        layer.W -= self.learning_rate * layer.vdW_corrected / np.sqrt(layer.sdW_corrected + AdaptiveMomentEstimation.EPSILON)
        layer.b -= self.learning_rate * layer.vdb_corrected / np.sqrt(layer.sdb_corrected + AdaptiveMomentEstimation.EPSILON)

        self.coefficient += 1
        self.learning_rate = self.lr_decay(self.learning_rate0)