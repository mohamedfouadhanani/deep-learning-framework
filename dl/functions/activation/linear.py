import numpy as np

from dl.functions import Function

class Linear(Function):
    def __call__(self, Z):
        return Z
    
    def prime(self, Z):
        return np.ones(Z.shape)