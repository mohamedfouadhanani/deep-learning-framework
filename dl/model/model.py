import sys
import numpy as np
import dill


class Model:
    def __init__(self, layers=None):
        self.layers = layers

    def compile(self, optimizer, loss):
        # settings optimizer
        self.optimizer = optimizer
        self.optimizer.model = self

        # settings loss
        self.loss = loss
        self.loss.model = self

        # settings n_units of different layers
        L = len(self.layers)
        for l in range(1, L):
            n_units = self.layers[l - 1].n_units
            self.layers[l].initialize(n_units)

        # optimizer initialization
        self.optimizer.initialize()

    def __call__(self, X, is_optimizing=False):
        A = X
        for layer in self.layers[1:]:
            A = layer(A, is_optimizing=is_optimizing)

        return A

    def append(self, layer):
        self.layers.append(layer)

    def optimize(self, X, y, n_epochs, verbose=True):
        history = self.optimizer(X, y, n_epochs, verbose)
        return history
    
    def parameters(self, including_biases=True):
        params = []
        for layer in self.layers[1:]:
            params.extend(layer.W.reshape(-1, ))
            if including_biases:
                params.extend(layer.b.reshape(-1, ))
            
        return params
    
    def save(self, file_path):
        pickle_model = dill.dumps(self)
        with open(f"{file_path}.dl", "wb") as file:
            file.write(pickle_model)

    @staticmethod
    def load(file_path):
        model = None
        try:
            with open(file_path, "rb") as dill_file:
                model = dill.load(dill_file)

            return model

        except FileNotFoundError as exception:
            print(exception)
            sys.exit(1)

    def __repr__(self):
        representation = ""

        # LAYERS
        representation += "Architecture\n"
        for layer in self.layers:
            representation += f"\t{layer}\n"
        

        representation += "\n"
        # LOSS
        representation += "Loss\n"
        representation += f"\t{self.loss}\n"

        representation += "\n"
        # OPTIMIZER
        representation += "Optimizer\n"
        representation += f"\t{self.optimizer}"

        return representation
