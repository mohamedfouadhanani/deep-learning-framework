import numpy as np


class StochasticGradientDescent:
    def __init__(self, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, X, y, n_epochs, verbose=True):
        history = {"loss": []}
        n, m = X.shape

        for epoch in range(n_epochs):
            # applying learning rate decay
            self.learning_rate = self.lr_decay(self.learning_rate0, epoch)

            for i in range(m):
                X_i = X[:, i].reshape(-1, 1)
                y_i = y[:, i].reshape(-1, 1)

                # forward propagation
                self.model.layers[0].A = X_i
                self.model.forward_propagation()

                # loss calculation
                y_hat = self.model.layers[-1].A
                loss = self.model.loss(y_hat, y_i)

                if self.model.loss.regularizer is not None:
                    for l in range(1, self.model.L):
                        loss += self.model.loss.regularizer(m, self.model.layers[l].W)

                # backward propagation - computing dWi & dbi for every layer
                self.model.layers[-1].dA = self.model.loss.prime(y_hat, y_i)
                self.model.backward_propagation(m)

                # updating dWi & dbi
                for l in range(1, self.model.L):
                    self.model.layers[l].W -= self.learning_rate * self.model.layers[l].dW
                    self.model.layers[l].b -= self.learning_rate * self.model.layers[l].db

                # printing to the console
                if verbose:
                    print(f"[{epoch + 1}/{n_epochs}, {i + 1}/{m}]: loss = {loss}")

            # keeping history
            history["loss"].append(loss)

        return history
