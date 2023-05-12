from dlf.activation_functions.activation_function import ActivationFunction

class ReLU(ActivationFunction):
    def __init__(self) -> None:

        def f(inputs):
            outputs = (inputs > 0) * inputs
            return outputs

        def df(inputs):
            outputs = (inputs > 0)
            return outputs
        
        super().__init__(f, df)