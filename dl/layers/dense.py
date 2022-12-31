import numpy as np

from dl.functions.activation import Linear
from dl.automatic_gradient.variable import Variable

class Dense:
    def __init__(self, n_units, activation_function=Linear):
        self.n_units = n_units
        self.activation_function = activation_function

    def initialize(self, n_inputs):
        self.n_inputs = n_inputs

        W = np.random.randn(self.n_units, self.n_inputs) * np.sqrt(1 / n_inputs)
        b = np.zeros((1, self.n_units))

        self.W = Variable.from_numpy(W)
        self.b = Variable.from_numpy(b)

    def __repr__(self):
        return f"Dense(n_units={self.n_units}, activation_function={self.activation_function.__name__})"
