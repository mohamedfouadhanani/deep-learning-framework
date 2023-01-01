import numpy as np
from decimal import Decimal

from dl.functions.activation.activation_function import ActivationFunction
from dl.automatic_gradient.variable import Variable

class ReLU(ActivationFunction):
    @staticmethod
    def run(variable: Variable) -> Variable:
        t: Decimal = max(Decimal(0), variable.data)

        output: Variable = Variable(
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