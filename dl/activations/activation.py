from dl.layer import Layer

class Activation(Layer):
    def __init__(self, f, df):
        cache = {}
        cache["inputs"] = None
        cache["outputs"] = None

        super().__init__(False, cache)
        
        self.f = f
        self.df = df

    def forward(self, inputs, is_optimizing):
        self.cache["inputs"] = inputs
        self.cache["outputs"] = self.f(self.cache["inputs"])
        return self.cache["outputs"]
    
    def backward(self, doutputs):
        self.cache["dinputs"] = doutputs * self.df(self.cache["inputs"])
        return self.cache["dinputs"]