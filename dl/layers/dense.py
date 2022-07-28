import numpy as np

from dl.functions.activation import Linear


class Dense:
    ID = 0

    def __init__(self, n_units, activation_function=Linear(), dropout=None):
        self.n_units = n_units
        self.activation_function = activation_function
        self.dropout = dropout
        self.ID = Dense.ID

        Dense.ID += 1

    def initialize(self, n_inputs):
        self.n_inputs = n_inputs

        self.W = np.random.randn(self.n_units, self.n_inputs) * np.sqrt(1 / n_inputs)
        self.b = np.zeros((self.n_units, 1))

    def __repr__(self):
        return f"\tDense_Layer_{self.ID}:\n\t\tW shape: {self.W.shape}\n\t\tb shape: {self.b.shape}\n\t\toutput shape: ({self.n_units}, 1)\n\t\tactivation function: {self.activation_function.__class__.__name__}\n\t\t# params: {self.W.size + self.b.size}"
