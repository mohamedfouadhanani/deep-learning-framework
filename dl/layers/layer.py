from dl.functions.activation.activation_function import ActivationFunction

class Layer:
    def __init__(self, n_units: int, activation_function: ActivationFunction):
        self.W = None
        self.b = None

    def initialize(self, n_inputs: int):
        pass

    def __call__(self, A):
        pass

    def __repr__(self):
        return f"Dense(n_units={self.n_units}, activation_function={self.activation_function.__name__})"
