import numpy as np
from dl.optimizers.optimizer import Optimizer

from dl import Dense

class MiniBatchGradientDescent(Optimizer):
    def __init__(self, batch_size, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, inputs, outputs, n_epochs, verbose=True):
        history = {"losses": []}
        m, _ = inputs.shape

        number_batches = np.ceil(m / self.batch_size).astype(int)

        for epoch in range(n_epochs):
            l = 0

            for t in range(number_batches):
                starting_index = t * self.batch_size
                finishing_index = starting_index + self.batch_size

                finishing_index = min(finishing_index, m)

                inputs_t = inputs[starting_index:finishing_index, :]
                outputs_t = outputs[starting_index:finishing_index, :]
                
                # forward propagation
                predictions = self.model(inputs_t)

                # compute loss
                l += self.model.loss.forward(predictions, outputs_t)

                # backward propagations
                dpredictions = self.model.loss.backward(predictions, outputs_t)
                self.model.backward_propagation(dpredictions)

                # parameters update
                for layer in self.model.layers:
                    if isinstance(layer, Dense):
                        layer.W -= self.learning_rate * layer.dW
                        layer.b -= self.learning_rate * layer.db

            average_loss = l / m
            if verbose:
                print(f"[{epoch + 1}/{n_epochs}]: loss = {average_loss}")
                
            history["losses"].append(average_loss)
            
        return history