from decimal import Decimal

from dl.optimizers.optimizer import Optimizer

class BatchGradientDescent(Optimizer):
    def __init__(self, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, X, y, n_epochs, verbose=True):
        history = {"loss": []}

        for epoch in range(n_epochs):
            # applying learning rate decay
            self.learning_rate = self.lr_decay(self.learning_rate0, epoch)

            # forward propagation
            y_pred = self.model(X, is_optimizing=True)

            # loss calculation
            loss = self.model.loss(y_pred, y)

            # model params
            params = self.model.parameters()

            # backward propagation - computing dWi & dbi for every layer
            self.zero_gradients(params)
            loss.backward()

            # updating dWi & dbi
            for param in params:
                self.step(param)

            # printing to the console
            if verbose:
                print(f"[{epoch + 1}/{n_epochs}]: loss = {loss}")

            # keeping history
            history["loss"].append(loss)

        return history
    
    def step(self, variable):
        variable.data -= Decimal(self.learning_rate) * variable.gradient

    def __repr__(self) -> str:
        return f"BatchGradientDescent(learning_rate={self.learning_rate0})"