from dl.activations.activation import Activation

class ReLU(Activation):
    def __init__(self):
        def f(inputs):
            outputs = (inputs > 0) * inputs
            return outputs

        def df(inputs):
            outputs = (inputs > 0)
            return outputs
        
        super().__init__(f, df)