import numpy as np
from dl.dense import Dense

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
    
    def optimize(self, inputs_train, outputs_train, n_epochs, verbose=True, inputs_val=None, outputs_val=None):
        history = {"train_losses": [], "val_losses": []}
        m, _ = inputs_train.shape

        n_batches = np.ceil(m / self.optimizer.batch_size).astype(int)
        
        for epoch in range(n_epochs):
            average_loss = 0
            for batch in range(n_batches):
                starting_index = batch * self.optimizer.batch_size
                finishing_index = min(starting_index + self.optimizer.batch_size, m)

                inputs_t = inputs_train[starting_index:finishing_index, :]
                outputs_t = outputs_train[starting_index:finishing_index, :]
                
                # forward propagation
                predictions = self.forward_propagation(inputs_t)

                # compute loss
                average_loss += self.loss.forward(predictions, outputs_t)

                # backward propagations
                dpredictions = self.loss.backward(predictions, outputs_t)
                self.backward_propagation(dpredictions)

                # parameters update
                for layer in self.layers:
                    if isinstance(layer, Dense):
                        self.optimizer(layer)
        
            average_loss /= m
                        
            history["train_losses"].append(average_loss)
            if inputs_val is not None and outputs_val is not None:
                val_predictions = self.forward_propagation(inputs_val, is_optimizing=False)
                val_l = self.loss.forward(val_predictions, outputs_val)
                history["val_losses"].append(val_l)

            if verbose:
                print(f"[{epoch + 1}/{n_epochs}]: loss = {average_loss}")
        
        return history