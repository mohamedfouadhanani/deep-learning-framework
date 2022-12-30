import numpy as np
import decimal
from decimal import Decimal
from typing import Callable

import dl.environment as environment
from dl.automatic_gradient.utils import build_topology

# SETUP
decimal.setcontext(decimal.Context(prec=environment.DECIMAL_PRECISION))

class Variable:
    def __init__(self, data, children=(), operation="", label=""):
        self.data: Decimal = Decimal(data)
        self.gradient: Decimal = Decimal(0.)
        self.previous: set["Variable"] = set(children)
        self.operation: str = operation
        self.label: str = label

        self._backward: Callable = lambda: None
    
    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"

    def __radd__(self, other) -> "Variable":
        return self + other
    
    def __iadd__(self, other) -> "Variable":
        return self + other
    
    def __rmul__(self, other) -> "Variable":
        return self * other
    
    def __neg__(self) -> "Variable":
        return self * -1
    
    def __sub__(self, other) -> "Variable":
        return self + -other
    
    def __rsub__(self, other) -> "Variable":
        return other + (-self)

    def __truediv__(self, other) -> "Variable":
        return self * other**-1
    
    def __rtruediv__(self, other) -> "Variable":
        return other * self**-1

    def __pow__(self, other) -> "Variable":
        assert isinstance(other, (int, float)), "only supporting integers & float at the moment"
        other = Decimal(other)
        t = self.data ** other
        
        output: Variable = Variable(
            data=t,
            children=(self, ),
            operation="**",
            label=f"({self.label})**{other}"
        )

        def _backward():
            self.gradient += other * self.data ** (other - 1) * output.gradient
        output._backward = _backward
        return output
    
    def __add__(self, other) -> "Variable":
        other = other if isinstance(other, Variable) else Variable(data=other)

        output = Variable(
            data=self.data + other.data, 
            children=(self, other), 
            operation="+",
            label=f"{self.label} + {other.label}"
        )

        def _backward():
            self.gradient += Decimal(1.0) * output.gradient
            other.gradient += Decimal(1.0) * output.gradient
        output._backward = _backward
        
        return output
    
    def __mul__(self, other) -> "Variable":
        other = other if isinstance(other, Variable) else Variable(data=other)

        output = Variable(
            data=self.data * other.data, 
            children=(self, other), 
            operation="*",
            label=f"{self.label}*{other.label}"
        )

        def _backward():
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient
        output._backward = _backward
        return output
    
    def backward(self) -> None:
        topology = []
        
        build_topology(self, topology)

        self.gradient = Decimal(1.0)
        for node in reversed(topology):
            node._backward()
    
    @staticmethod
    def from_numpy(ndarray):
        ndarray = ndarray.astype(object)
        h, w = ndarray.shape

        for i in range(h):
            for j in range(w):
                ndarray[i, j] = Variable(data=ndarray[i, j])
            
        return ndarray