import sys
import numpy as np
import dill


class Model:
    def __init__(self, layers=None):
        self.layers = layers
        self.L = len(self.layers)

    def __call__(self, A0):
        A = A0
        for l in range(1, self.L):
            Z = np.dot(self.layers[l].W, A) + self.layers[l].b
            A = self.layers[l].activation_function(Z)

        return A

    def compile(self, optimizer, loss):
        # settings optimizer
        self.optimizer = optimizer
        self.optimizer.model = self

        # settings loss
        self.loss = loss
        self.loss.model = self

        # settings n_units of different layers
        for l in range(1, self.L):
            n_units = self.layers[l - 1].n_units
            self.layers[l].initialize(n_units)

        # optimizer initialization
        self.optimizer.initialize()

    def optimize(self, X, y, n_epochs, verbose=True):
        history = self.optimizer(X, y, n_epochs, verbose)
        return history

    def forward_propagation(self):
        for l in range(1, self.L):
            self.layers[l].Z = np.dot(
                self.layers[l].W, self.layers[l - 1].A) + self.layers[l].b
            self.layers[l].A = self.layers[l].activation_function(self.layers[l].Z)

            if self.layers[l].dropout is not None:
                self.layers[l].A = self.layers[l].dropout(self.layers[l].A)

    def backward_propagation(self, m):
        for l in range(self.L - 1, 0, -1):
            # applying dropout
            if self.layers[l].dropout is not None:
                self.layers[l].dA = self.layers[l].dropout(self.layers[l].dA)

            # dZi = dAi * g[l]'(Zi)
            self.layers[l].dZ = self.layers[l].dA * \
                self.layers[l].activation_function.prime(self.layers[l].Z)
            # dWi = (1 / m) * dZi * Ai-1.T
            self.layers[l].dW = (1 / m) * np.dot(self.layers[l].dZ, self.layers[l - 1].A.T)
            if self.loss.regularizer is not None:
                self.layers[l].dW += self.loss.regularizer.prime(m, self.layers[l].W)
            # dbi = (1 / m) * np.sum(dZi, axis=1, keepdims=True)
            self.layers[l].db = (1 / m) * np.sum(self.layers[l].dZ, axis=1, keepdims=True)
            # dAi-1 = Wi.T * dZi
            self.layers[l - 1].dA = np.dot(self.layers[l].W.T, self.layers[l].dZ)

    def save(self, file_path):
        # pickle the model
        pickle_model = dill.dumps(self)
        with open(f"{file_path}.dl", "wb") as file:
            file.write(pickle_model)

    @staticmethod
    def load(file_path):
        model = None
        try:
            file = open(file_path, "rb")
            pickle_model = file.read()
            model = dill.loads(pickle_model)
            file.close()
            return model

        except FileNotFoundError as exception:
            print(exception)
            sys.exit(1)

    def summary(self):
        print("Layers")
        for layer in self.layers:
            print(layer)

        print(f"Loss\n\t{self.loss.__class__.__name__}\n\t\tregularizer: {self.loss.regularizer.__class__.__name__ if self.loss.regularizer is not None else 'None'}")
        print(f"Optimizer\n\t{self.optimizer.__class__.__name__}")
