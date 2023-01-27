from dl.layer import Layer

class Activation(Layer):
    def __init__(self, f, df):
        self.f = f
        self.df = df

    def forward(self, inputs, is_optimizing):
        self.inputs = inputs
        self.outputs = self.f(self.inputs)
        return self.outputs
    
    def backward(self, doutputs):
        self.dinputs = doutputs * self.df(self.inputs)
        return self.dinputs