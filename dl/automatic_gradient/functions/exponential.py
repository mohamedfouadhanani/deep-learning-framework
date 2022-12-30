import numpy as np

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class Exponential(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:    
        t = variable.data.exp()

        output = Variable(
            data=t, 
            children=(variable, ),
            operation="exp",
            label=f"exp({variable.label})"
        )

        def _backward():
            variable.gradient += t * output.gradient
        output._backward = _backward
        return output

Exponential.run = np.vectorize(Exponential.run)