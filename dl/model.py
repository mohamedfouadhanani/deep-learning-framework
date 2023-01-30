import numpy as np
import dill

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.history = {
            "train_losses": [],
            "val_losses": []
        }
    
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
        self.optimizer = optimizer
    
    def optimize(self, inputs_train, outputs_train, n_epochs, verbose=True, inputs_val=None, outputs_val=None):
        m, _ = inputs_train.shape

        n_batches = np.ceil(m / self.optimizer.batch_size).astype(int)
        
        for epoch in range(n_epochs):
            for batch in range(n_batches):
                starting_index = batch * self.optimizer.batch_size
                finishing_index = min(starting_index + self.optimizer.batch_size, m)

                inputs_t = inputs_train[starting_index:finishing_index, :]
                outputs_t = outputs_train[starting_index:finishing_index, :]
                
                # forward propagation
                predictions = self.forward_propagation(inputs_t)

                # compute loss
                # batch_loss = self.loss.forward(predictions, outputs_t)

                # backward propagations
                dpredictions = self.loss.backward(predictions, outputs_t)
                self.backward_propagation(dpredictions)

                # parameters update
                for layer in self.layers:
                    if layer.is_trainable:
                        self.optimizer(layer)
                        

            training_predictions = self.forward_propagation(inputs_train, is_optimizing=False)
            training_loss = self.loss.forward(training_predictions, outputs_train)
            self.history["train_losses"].append(training_loss)

            if inputs_val is not None and outputs_val is not None:
                val_predictions = self.forward_propagation(inputs_val, is_optimizing=False)
                val_l = self.loss.forward(val_predictions, outputs_val)
                self.history["val_losses"].append(val_l)

            if verbose:
                print(f"[{epoch + 1}/{n_epochs}]: Training Loss: {training_loss} {f', Validation Loss: {val_l}' if inputs_val is not None and outputs_val is not None else ''}")
        
        return self.history
    
    def save(self, file_path, silent=False):
        try:
            with open(file_path, "wb") as dill_file:
                dill.dump(self, dill_file)
            return True
        except Exception as e:
            if not silent:
                print(e)
            return False
    
    @staticmethod
    def load(file_path, silent=False):
        model = None
        try:
            with open(file_path, "rb") as dill_file:
                model = dill.load(dill_file)
        except Exception as e:
            if not silent:
                print(e)
        finally:
            return model