from dl.optimizers.optimizer import Optimizer

from dl import Dense

class BatchGradientDescent(Optimizer):
    def __init__(self, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, inputs, outputs, n_epochs, verbose=True):
        history = {"losses": []}

        for epoch in range(n_epochs):
            # forward propagation
            predictions = self.model(inputs)

            # compute loss
            l = self.model.loss.forward(predictions, outputs)
            history["losses"].append(l)

            if verbose:
                print(f"[{epoch + 1}/{n_epochs}]: loss = {l}")

            # backward propagations
            dpredictions = self.model.loss.backward(predictions, outputs)
            self.model.backward_propagation(dpredictions)

            # parameters update
            for layer in self.model.layers:
                if isinstance(layer, Dense):
                    layer.W -= self.learning_rate * layer.dW
                    layer.b -= self.learning_rate * layer.db
            
        return history