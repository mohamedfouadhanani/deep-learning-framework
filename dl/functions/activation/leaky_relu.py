import numpy as np
from decimal import Decimal

from dl.functions.activation.activation_function import ActivationFunction
from dl.automatic_gradient.variable import Variable

class LeakyReLU(ActivationFunction):
    alpha: Decimal = Decimal(0.01)

    @staticmethod
    def run(variable: Variable) -> Variable:
        t: Decimal = max(LeakyReLU.alpha * variable.data, variable.data)

        output: Variable = Variable(
            data=t,
            children=(variable, ),
            operation="leaky_relu",
            label=f"leaky_relu({variable.label})"
        )

        def _backward():
            variable.gradient += (Decimal(1.0) if t > 0 else LeakyReLU.alpha) * output.gradient
        output._backward = _backward

        return output

LeakyReLU.run = np.vectorize(LeakyReLU.run)