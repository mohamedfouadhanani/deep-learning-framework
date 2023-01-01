import numpy as np

from dl.functions.activation import Linear
from dl.automatic_gradient.variable import Variable
from dl.functions.activation.activation_function import ActivationFunction
from dl.layers.layer import Layer

class Dense(Layer):
    def __init__(self, n_units: int, activation_function: ActivationFunction=Linear):
        self.n_units = n_units
        self.activation_function = activation_function
        self.W = None
        self.b = None

    def initialize(self, n_inputs: int):
        self.n_inputs = n_inputs

        W = np.random.randn(self.n_units, self.n_inputs) * np.sqrt(1 / n_inputs)
        b = np.zeros((1, self.n_units))

        self.W = Variable.from_numpy(W)
        self.b = Variable.from_numpy(b)

    def __call__(self, A, **kwargs):
        self.Z = np.dot(A, self.W.T) + self.b
        self.A = self.activation_function.run(self.Z)

        return self.A

    def __repr__(self):
        return f"Dense(n_units={self.n_units}, activation_function={self.activation_function.__name__})"
