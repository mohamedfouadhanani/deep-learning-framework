from dl.optimizers.optimizer import Optimizer

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, batch_size, lr_decay=lambda lr0: lr0):
        super().__init__(learning_rate, batch_size, lr_decay)
    
    def __call__(self, layer):
        for param in layer.params:
            layer.cache[param] -= self.learning_rate * layer.cache[f"d{param}"]

        self.learning_rate = self.lr_decay(self.learning_rate0)