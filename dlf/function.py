from abc import ABC, abstractmethod

class Function(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
    
    @abstractmethod
    def backward(self, inputs):
        pass