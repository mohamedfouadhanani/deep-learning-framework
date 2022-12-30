import numpy as np
from decimal import Decimal

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class TanH(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:
        ex = variable.data.exp()
        enx = (-variable.data).exp()
        t = (ex - enx) / (ex + enx)

        output = Variable(
            data=t, 
            children=(variable, ),
            operation="tanh",
            label=f"tanh({variable.label})"
        )

        def _backward():
            variable.gradient += (Decimal(1) - t ** Decimal(2)) * output.gradient
        output._backward = _backward
        return output

TanH.run = np.vectorize(TanH.run)