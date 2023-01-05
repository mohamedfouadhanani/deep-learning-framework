import numpy as np
import fractions
from decimal import Decimal

from dl.layers.dense import Dense
from dl.optimizers.optimizer import Optimizer

class RMSProp(Optimizer):
    EPSILON = 1e-7

    def __init__(self, batch_size, learning_rate, lr_decay=lambda lr0, epoch: lr0, beta=0.999):
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.beta = beta

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.sdW = np.zeros(layer.W.shape)
                layer.sdb = np.zeros(layer.b.shape)

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
                self.step()

                # printing to the console
                if verbose:
                    print(f"[{epoch + 1}/{n_epochs}, {t + 1}/{number_batches}]: loss = {loss.data}")
                
                average_loss += loss.data

            # keeping history
            history["loss"].append(average_loss / number_batches)

        return history
    
    def step(self):
        for layer in self.model.layers:
            if not isinstance(layer, Dense):
                continue
            
            # updating sdW
            n_units, n_inputs = layer.sdW.shape
            for i in range(n_units):
                for j in range(n_inputs):
                    layer.sdW[i, j] = Decimal(self.beta) * Decimal(layer.sdW[i, j]) + Decimal(1 - self.beta) * layer.W[i, j].gradient ** 2
                    layer.W[i, j].data -= Decimal(self.learning_rate) * layer.W[i, j].gradient / (Decimal(layer.sdW[i, j]).sqrt() + Decimal(RMSProp.EPSILON))
            
            # updating sdb
            n_units, n_inputs = layer.sdb.shape
            for i in range(n_units):
                for j in range(n_inputs):
                    layer.sdb[i, j] = Decimal(self.beta) * Decimal(layer.sdb[i, j]) + Decimal(1 - self.beta) * layer.b[i, j].gradient ** 2
                    layer.b[i, j].data -= Decimal(self.learning_rate) * layer.b[i, j].gradient / (Decimal(layer.sdb[i, j]).sqrt() + Decimal(RMSProp.EPSILON))

    def __repr__(self) -> str:
        return f"RMSProp(batch_size={self.batch_size}, learning_rate={self.learning_rate0}, beta={self.beta})"