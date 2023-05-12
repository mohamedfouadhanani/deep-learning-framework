from dlf.layers.layer import Layer

class ActivationFunction(Layer):

    def __init__(self, f, df) -> None:
        super().__init__()
        
        self.f = f
        self.df = df
    
    def forward(self, inputs, is_optimizing=True):
        self.inputs = inputs
        self.outputs = self.f(self.inputs)
        return self.outputs
    
    def backward(self, doutputs):
        self.doutputs = doutputs
        self.dinputs = doutputs * self.df(self.inputs)
        return self.dinputs