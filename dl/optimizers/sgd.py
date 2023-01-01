from decimal import Decimal

from dl.optimizers.optimizer import Optimizer

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, X, y, n_epochs, verbose=True):
        history = {"loss": []}
        m, n = X.shape

        for epoch in range(n_epochs):
            # applying learning rate decay
            self.learning_rate = self.lr_decay(self.learning_rate0, epoch)

            average_loss = 0
            for i in range(m):
                X_i = X[i, :].reshape(-1, 1)
                y_i = y[i, :].reshape(-1, 1)

                # forward propagation
                y_pred = self.model(X_i, is_optimizing=True)

                # loss calculation
                loss = self.model.loss(y_pred, y_i)

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
                    print(f"[{epoch + 1}/{n_epochs}, {i + 1}/{m}]: loss = {loss.data}")
                
                average_loss += loss.data

            # keeping history
            history["loss"].append(average_loss)

        return history

    def step(self, variable):
        variable.data -= Decimal(self.learning_rate) * variable.gradient

    def __repr__(self) -> str:
        return f"StochasticGradientDescent(learning_rate={self.learning_rate0})"