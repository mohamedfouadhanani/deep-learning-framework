import numpy as np

from dlf.optimizers.optimizer import Optimizer
from dlf.trainable import Trainable
from dlf.layers.trainable_layer import TrainableLayer

class RMSProp(Optimizer):
    EPSILON = 1e-7
    
    def __init__(self, learning_rate, beta = 0.999) -> None:
        super().__init__(learning_rate)
        self.beta = beta

    def on_training_start(self, trainer):
        for layer in trainer.model.layers:
            if not isinstance(layer, Trainable):
                continue
            
            if not layer.is_trainable:
                continue
            
            for param in layer.params:
                layer.cache[f"sd{param}"] = np.zeros_like(layer.cache[param])

    def update(self, layer: TrainableLayer):
        for param in layer.params:
            layer.cache[f"sd{param}"] = self.beta * layer.cache[f"sd{param}"] + (1 - self.beta) * layer.cache[f"d{param}"] ** 2
            layer.cache[param] -= self.learning_rate * layer.cache[f"d{param}"] / np.sqrt(layer.cache[f"sd{param}"] + RMSProp.EPSILON)
