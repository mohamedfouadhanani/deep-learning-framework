import numpy as np


class MiniBatchGradientDescent:
    def __init__(self, batch_size, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, X, y, n_epochs, verbose=True):
        history = {"loss": []}
        n, m = X.shape
        number_batches = np.ceil(m / self.batch_size).astype(int)

        for epoch in range(n_epochs):
            # applying learning rate decay
            self.learning_rate = self.lr_decay(self.learning_rate0, epoch)

            for t in range(number_batches):
                starting_index = t * self.batch_size
                finishing_index = starting_index + self.batch_size

                finishing_index = min(finishing_index, m)

                X_t = X[:, starting_index:finishing_index]
                y_t = y[:, starting_index:finishing_index]

                # forward propagation
                self.model.layers[0].A = X_t
                self.model.forward_propagation()

                # loss calculation
                y_hat = self.model.layers[-1].A
                loss = self.model.loss(y_hat, y_t)

                if self.model.loss.regularizer is not None:
                    for l in range(1, self.model.L):
                        loss += self.model.loss.regularizer(m, self.model.layers[l].W)

                # backward propagation - computing dWi & dbi for every layer
                self.model.layers[-1].dA = self.model.loss.prime(y_hat, y_t)
                self.model.backward_propagation(m)

                # updating dWi & dbi
                for l in range(1, self.model.L):
                    self.model.layers[l].W -= self.learning_rate * self.model.layers[l].dW
                    self.model.layers[l].b -= self.learning_rate * self.model.layers[l].db

                # printing to the console
                if verbose:
                    print(f"[{epoch + 1}/{n_epochs}, {t + 1}/{number_batches}]: loss = {loss}")

            # keeping history
            history["loss"].append(loss)

        return history
