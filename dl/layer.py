class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs, is_optimizing):
        pass

    def backward(self, doutputs):
        pass