from typing import List

import dill

from dlf.layers.layer import Layer

class Sequential(Layer):
    
    def __init__(self, layers: List[Layer]) -> None:
        super().__init__()
        self.layers = layers
    
    def append(self, layer: Layer):
        self.layers.append(layer)
        
    def forward(self, inputs, is_optimizing=True):
        self.inputs = inputs
        
        outputs = inputs.copy()
        for layer in self.layers:
            outputs = layer.forward(outputs, is_optimizing=is_optimizing)
        
        self.outputs = outputs
        
        return self.outputs
    
    def backward(self, doutputs):
        self.doutputs = doutputs
        
        dinputs = doutputs.copy()
        for layer in reversed(self.layers):
            dinputs = layer.backward(dinputs)
            
        self.dinputs = dinputs
        
        return self.dinputs
    
    def save(self, file_path, silent=False):
        try:
            with open(file_path, "wb") as dill_file:
                dill.dump(self, dill_file)
            return True
        except Exception as e:
            if not silent:
                print(e)
            return False
    
    @staticmethod
    def load(file_path, silent=False):
        model = None
        try:
            with open(file_path, "rb") as dill_file:
                model = dill.load(dill_file)
        except Exception as e:
            if not silent:
                print(e)
        finally:
            return model