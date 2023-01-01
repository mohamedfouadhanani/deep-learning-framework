import numpy as np
from decimal import Decimal

from dl.optimizers.optimizer import Optimizer

class MiniBatchGradientDescent(Optimizer):
    def __init__(self, batch_size, learning_rate, lr_decay=lambda lr0, epoch: lr0):
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def initialize(self):
        pass

    def __call__(self, X, y, n_epochs, verbose=True):
        history = {"loss": []}

        m, n = X.shape
        number_batches = np.ceil(m / self.batch_size).astype(int)

        for epoch in range(n_epochs):
            # applying learning rate decay
            self.learning_rate = self.lr_decay(self.learning_rate0, epoch)

            average_loss = 0
            for t in range(number_batches):
                starting_index = t * self.batch_size

                finishing_index = starting_index + self.batch_size
                finishing_index = min(finishing_index, m)

                X_t = X[starting_index:finishing_index, :]
                y_t = y[starting_index:finishing_index, :]

                # forward propagation
                y_pred = self.model(X_t, is_optimizing=True)

                # loss calculation
                loss = self.model.loss(y_pred, y_t)

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
                    print(f"[{epoch + 1}/{n_epochs}, {t + 1}/{number_batches}]: loss = {loss.data}")
                
                average_loss += loss.data

            # keeping history
            history["loss"].append(average_loss / number_batches)

        return history
    
    def step(self, variable):
        variable.data -= Decimal(self.learning_rate) * variable.gradient

    def __repr__(self) -> str:
        return f"MiniBatchGradientDescent(batch_size={self.batch_size}, learning_rate={self.learning_rate0})"
