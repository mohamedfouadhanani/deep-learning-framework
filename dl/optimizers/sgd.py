from dl.optimizers.optimizer import Optimizer

from dl import Dense

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, inputs, outputs, n_epochs, verbose=True):
        history = {"losses": []}
        m, _ = inputs.shape

        for epoch in range(n_epochs):
            l = 0
            for i in range(m):
                inputs_i = inputs[i, :].reshape(-1, 1)
                outputs_i = outputs[i, :].reshape(-1, 1)
                
                # forward propagation
                predictions = self.model(inputs_i)

                # compute loss
                l += self.model.loss.forward(predictions, outputs_i)

                # backward propagations
                dpredictions = self.model.loss.backward(predictions, outputs_i)
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