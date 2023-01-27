class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, inputs, is_optimizing=True):
        outputs = self.forward_propagation(inputs, is_optimizing)
        return outputs
    
    def forward_propagation(self, inputs, is_optimizing=True):
        self.outputs = inputs
        for layer in self.layers:
            self.outputs = layer.forward(self.outputs, is_optimizing=is_optimizing)
        
        return self.outputs
    
    def backward_propagation(self, doutputs):
        self.dinputs = doutputs
        for layer in self.layers[::-1]:
            self.dinputs = layer.backward(self.dinputs)
        
        return self.dinputs
    
    def compile(self, loss, optimizer):
        self.loss = loss
        self.loss.model = self
        
        self.optimizer = optimizer
        self.optimizer.model = self

        self.optimizer.initialize()
    
    def optimize(self, inputs, outputs, n_epochs, verbose=True):
        history = self.optimizer(inputs, outputs, n_epochs, verbose)
        return history