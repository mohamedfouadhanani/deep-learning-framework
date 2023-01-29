class Layer:
    def __init__(self, is_trainable=False, cache=None, params=None):
        self.is_trainable = is_trainable
        self.cache = cache
        self.params = params
    
    def forward(self, inputs, is_optimizing):
        pass

    def backward(self, doutputs):
        pass