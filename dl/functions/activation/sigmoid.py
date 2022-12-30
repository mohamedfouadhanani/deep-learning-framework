import numpy as np
from decimal import Decimal

from dl.functions.activation.activation_function import ActivationFunction
from dl.automatic_gradient.variable import Variable

class Sigmoid(ActivationFunction):
    @staticmethod
    def run(variable):
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