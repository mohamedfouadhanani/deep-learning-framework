from abc import ABC, abstractmethod

class Initializer(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def initialize(self, shape = None):
        pass