from abc import abstractmethod
from dlf.function import Function

class Layer(Function):
    
    def __init__(self) -> None:
        self.inputs = None
        self.dinputs = None
        
        self.outputs = None
        self.doutputs = None
        
        self.cache = {}
    
    @abstractmethod
    def forward(self, inputs, is_optimizing=True):
        pass
    
    @abstractmethod
    def backward(self, doutputs):
        pass