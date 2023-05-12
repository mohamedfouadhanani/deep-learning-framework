import numpy as np

from dlf.optimizers.optimizer import Optimizer
from dlf.trainable import Trainable
from dlf.layers.trainable_layer import TrainableLayer

class AdaptiveMomentEstimation(Optimizer):
    EPSILON = 1e-7
    
    def __init__(self, learning_rate, beta1 = 0.9, beta2 = 0.99) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.coefficient = 0
    
    def on_training_start(self, trainer):
        for layer in trainer.model.layers:
            if not isinstance(layer, Trainable):
                continue
            
            if not layer.is_trainable:
                continue
            
            for param in layer.params:
                layer.cache[f"vd{param}"] = np.zeros_like(layer.cache[param])
                layer.cache[f"sd{param}"] = np.zeros_like(layer.cache[param])
                
                layer.cache[f"vd{param}_corrected"] = None
                layer.cache[f"sd{param}_corrected"] = None

    def update(self, layer: TrainableLayer):
        for param in layer.params:
            layer.cache[f"vd{param}"] = self.beta1 * layer.cache[f"vd{param}"] + (1 - self.beta1) * layer.cache[f"d{param}"]
            layer.cache[f"sd{param}"] = self.beta2 * layer.cache[f"sd{param}"] + (1 - self.beta2) * layer.cache[f"d{param}"] ** 2
            
            layer.cache[f"vd{param}_corrected"] = layer.cache[f"vd{param}"] / (1 - self.beta1 ** (self.coefficient + 1))
            layer.cache[f"sd{param}_corrected"] = layer.cache[f"sd{param}"] / (1 - self.beta2 ** (self.coefficient + 1))
            
            layer.cache[param] -= self.learning_rate * layer.cache[f"vd{param}_corrected"] / np.sqrt(layer.cache[f"sd{param}_corrected"] + AdaptiveMomentEstimation.EPSILON)
    
    def on_epoch_finish(self, trainer):
        self.coefficient += 1