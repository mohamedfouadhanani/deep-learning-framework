import numpy as np
from decimal import Decimal

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class Sigmoid(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:    
        t = Decimal(1) / (Decimal(1) + (-variable.data).exp())

        output = Variable(
            data=t,
            children=(variable, ),
            operation="sigmoid",
            label=f"sigmoid({variable.label})"
        )

        def _backward():
            variable.gradient += t * (Decimal(1) - t) * output.gradient
        output._backward = _backward

        return output

Sigmoid.run = np.vectorize(Sigmoid.run)