from decimal import Decimal

class Optimizer:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def __call__(self, X, y, n_epochs: int, verbose: bool=True):
        pass

    def zero_gradients(self, params):
        for param in params:
            param.gradient = Decimal(0.)