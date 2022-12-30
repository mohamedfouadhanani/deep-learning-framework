import numpy as np
from decimal import Decimal

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class ReLU(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:
        t = max(Decimal(0), variable.data)

        output = Variable(
            data=t,
            children=(variable, ),
            operation="relu",
            label=f"relu({variable.label})"
        )

        def _backward():
            variable.gradient += (t > 0) * output.gradient
        output._backward = _backward

        return output

ReLU.run = np.vectorize(ReLU.run)