import numpy as np
from decimal import Decimal

from dl.layers.dense import Dense
from dl.optimizers.optimizer import Optimizer

class AdaptiveMomentEstimation(Optimizer):
    EPSILON = 1e-7

    def __init__(self, batch_size, learning_rate, lr_decay=lambda lr0, epoch: lr0, beta1=0.9, beta2=0.999):
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                # velocity
                layer.vdW = np.zeros(layer.W.shape)
                layer.vdb = np.zeros(layer.b.shape)
                
                # squares
                layer.sdW = np.zeros(layer.W.shape)
                layer.sdb = np.zeros(layer.b.shape)

                # velocity corrected
                layer.vdW_corrected = np.zeros(layer.W.shape)
                layer.vdb_corrected = np.zeros(layer.b.shape)

                # squares corrected
                layer.sdW_corrected = np.zeros(layer.W.shape)
                layer.sdb_corrected = np.zeros(layer.b.shape)

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
                self.step(t)

                # printing to the console
                if verbose:
                    print(f"[{epoch + 1}/{n_epochs}, {t + 1}/{number_batches}]: loss = {loss.data}")
                
                average_loss += loss.data

            # keeping history
            history["loss"].append(average_loss / number_batches)

        return history

    def step(self, t):
        for layer in self.model.layers:
            if not isinstance(layer, Dense):
                continue
            
            # updating vdW
            # updating vdW
            n_units, n_inputs = layer.vdW.shape
            for i in range(n_units):
                for j in range(n_inputs):
                    layer.vdW[i, j] = Decimal(self.beta1) * Decimal(layer.vdW[i, j]) + Decimal(1 - self.beta1) * layer.W[i, j].gradient
                    layer.sdW[i, j] = Decimal(self.beta2) * Decimal(layer.sdW[i, j]) + Decimal(1 - self.beta2) * layer.W[i, j].gradient ** 2
                    layer.vdW_corrected[i, j] = Decimal(layer.vdW[i, j]) / Decimal(1 - self.beta1 ** (t + 1))
                    layer.sdW_corrected[i, j] = Decimal(layer.sdW[i, j]) / Decimal(1 - self.beta2 ** (t + 1))

                    layer.W[i, j].data -= Decimal(self.learning_rate) * Decimal(layer.vdW_corrected[i, j]) / (Decimal(layer.sdW_corrected[i, j]) + Decimal(AdaptiveMomentEstimation.EPSILON)).sqrt()
            
            # updating vdb
            n_units, n_inputs = layer.vdb.shape
            for i in range(n_units):
                for j in range(n_inputs):
                    layer.vdb[i, j] = Decimal(self.beta1) * Decimal(layer.vdb[i, j]) + Decimal(1 - self.beta1) * layer.b[i, j].gradient
                    layer.sdb[i, j] = Decimal(self.beta2) * Decimal(layer.sdb[i, j]) + Decimal(1 - self.beta2) * layer.b[i, j].gradient ** 2
                    layer.vdb_corrected[i, j] = Decimal(layer.vdb[i, j]) / Decimal(1 - self.beta1 ** (t + 1))
                    layer.sdb_corrected[i, j] = Decimal(layer.sdb[i, j]) / Decimal(1 - self.beta2 ** (t + 1))

                    layer.b[i, j].data -= Decimal(self.learning_rate) * Decimal(layer.vdb_corrected[i, j]) / (Decimal(layer.sdb_corrected[i, j]) + Decimal(AdaptiveMomentEstimation.EPSILON)).sqrt()

    def __repr__(self) -> str:
        return f"AdaptiveMomentEstimation(batch_size={self.batch_size}, learning_rate={self.learning_rate0}, beta1={self.beta1}, beta2={self.beta2})"