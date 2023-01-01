import sys
import dill
from typing import List

from dl.regularization import Dropout
from dl.optimizers.optimizer import Optimizer
from dl.functions.loss.loss_function import LossFunction
from dl.layers.layer import Layer
from dl.automatic_gradient.variable import Variable

class Model:
    def __init__(self, layers: List[Layer]=None):
        self.layers = layers

    def compile(self, optimizer: Optimizer, loss: LossFunction):
        # settings optimizer
        self.optimizer = optimizer
        self.optimizer.model = self

        # settings loss
        self.loss = loss
        self.loss.model = self

        # settings n_units of different layers
        previous_layer: int = 0
        current_layer: int = 1

        while current_layer < len(self.layers):
            if isinstance(self.layers[current_layer], Dropout):
                current_layer += 1
                continue
            
            n_units: int = self.layers[previous_layer].n_units
            self.layers[current_layer].initialize(n_units)
            previous_layer = current_layer
            current_layer += 1
            

        # optimizer initialization
        self.optimizer.initialize()

    def __call__(self, X, is_optimizing=False):
        A = X
        for layer in self.layers[1:]:
            A = layer(A, is_optimizing=is_optimizing)

        return A

    def append(self, layer: Layer):
        self.layers.append(layer)

    def optimize(self, X, y, n_epochs: int, verbose: bool=True):
        history = self.optimizer(X, y, n_epochs, verbose)
        return history
    
    def parameters(self, including_biases: bool=True):
        params: List[Variable] = []
        for layer in self.layers[1:]:
            params.extend(layer.W.reshape(-1, ))
            if including_biases:
                params.extend(layer.b.reshape(-1, ))
            
        return params
    
    def save(self, file_path: str):
        pickle_model = dill.dumps(self)
        with open(f"{file_path}.dl", "wb") as file:
            file.write(pickle_model)

    @staticmethod
    def load(file_path: str) -> "Model":
        model = None
        try:
            with open(file_path, "rb") as dill_file:
                model = dill.load(dill_file)

            return model

        except FileNotFoundError as exception:
            print(exception)
            sys.exit(1)

    def __repr__(self) -> str:
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
