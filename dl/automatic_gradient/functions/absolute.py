import numpy as np
from decimal import Decimal

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class Absolute(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:
        t = abs(variable.data)

        output = Variable(
            data=t, 
            children=(variable, ),
            operation="abs",
            label=f"abs({variable.label})"
        )

        def _backward():
            variable.gradient += (Decimal(-1.0) if variable.data < 0 else Decimal(1.0)) * output.gradient
        output._backward = _backward
        return output

Absolute.run = np.vectorize(Absolute.run)