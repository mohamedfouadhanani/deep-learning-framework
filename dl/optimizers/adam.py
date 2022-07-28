import numpy as np


class AdaptiveMomentEstimation:
    EPSILON = 1e-7

    def __init__(self, batch_size, learning_rate, lr_decay=lambda lr0, epoch: lr0, beta1=0.9, beta2=0.999):
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2

    def initialize(self):
        for l in range(1, self.model.L):
            # velocity
            self.model.layers[l].vdW = np.zeros((self.model.layers[l].n_units, self.model.layers[l].n_inputs))
            self.model.layers[l].vdb = np.zeros((self.model.layers[l].n_units, 1))

            # squares
            self.model.layers[l].sdW = np.zeros((self.model.layers[l].n_units, self.model.layers[l].n_inputs))
            self.model.layers[l].sdb = np.zeros((self.model.layers[l].n_units, 1))

            # velocity corrected
            self.model.layers[l].vdW_corrected = None
            self.model.layers[l].vdb_corrected = None

            # squares corrected
            self.model.layers[l].sdW_corrected = None
            self.model.layers[l].sdb_corrected = None

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

                # backward propagation - computing dWi & dbi for every layer
                self.model.layers[-1].dA = self.model.loss.prime(y_hat, y_t)
                self.model.backward_propagation(m)

                # updating dWi & dbi
                for l in range(1, self.model.L):
                    # velocity update
                    self.model.layers[l].vdW = self.beta1 * self.model.layers[l].vdW + \
                        (1 - self.beta1) * self.model.layers[l].dW
                    self.model.layers[l].vdb = self.beta1 * self.model.layers[l].vdb + \
                        (1 - self.beta1) * self.model.layers[l].db

                    # squres update
                    self.model.layers[l].sdW = self.beta2 * self.model.layers[l].sdW + \
                        (1 - self.beta2) * self.model.layers[l].dW ** 2
                    self.model.layers[l].sdb = self.beta2 * self.model.layers[l].sdb + \
                        (1 - self.beta2) * self.model.layers[l].db ** 2

                    # velocity correction
                    self.model.layers[l].vdW_corrected = self.model.layers[l].vdW / (1 - self.beta1 ** (t + 1))
                    self.model.layers[l].vdb_corrected = self.model.layers[l].vdb / (1 - self.beta1 ** (t + 1))

                    # squares correction
                    self.model.layers[l].sdW_corrected = self.model.layers[l].sdW / (1 - self.beta2 ** (t + 1))
                    self.model.layers[l].sdb_corrected = self.model.layers[l].sdb / (1 - self.beta2 ** (t + 1))

                    self.model.layers[l].W -= self.learning_rate * self.model.layers[l].vdW_corrected / \
                        np.sqrt(self.model.layers[l].sdW + AdaptiveMomentEstimation.EPSILON)
                    self.model.layers[l].b -= self.learning_rate * self.model.layers[l].vdb_corrected / \
                        np.sqrt(self.model.layers[l].sdb + AdaptiveMomentEstimation.EPSILON)

                # printing to the console
                if verbose:
                    print(f"[{epoch + 1}/{n_epochs}, {t + 1}/{number_batches}]: loss = {loss}")

            # keeping history
            history["loss"].append(loss)

        return history
