import numpy as np
from decimal import Decimal

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class Log(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:
        t = variable.data.ln()

        output = Variable(
            data=t,
            children=(variable, ),
            operation="log",
            label=f"log({variable.label})"
        )

        def _backward():
            variable.gradient += Decimal(1.0) / variable.data * output.gradient
        output._backward = _backward

        return output

Log.run = np.vectorize(Log.run)