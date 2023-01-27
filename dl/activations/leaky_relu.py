from dl.activations.activation import Activation

class LeakyReLU(Activation):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        
        def f(inputs):
            outputs = ((inputs > 0) * inputs) + ((inputs <= 0) * self.alpha * inputs)
            return outputs

        def df(inputs):
            outputs = (inputs > 0) + ((inputs <= 0) * self.alpha)
            return outputs
        
        super().__init__(f, df)