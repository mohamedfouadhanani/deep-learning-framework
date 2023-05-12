from dlf.optimizers.optimizer import Optimizer
from dlf.layers.trainable_layer import TrainableLayer


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate) -> None:
        super().__init__(learning_rate)

    def update(self, layer: TrainableLayer):
        for param in layer.params:
            layer.cache[param] -= self.learning_rate * layer.cache[f"d{param}"]
