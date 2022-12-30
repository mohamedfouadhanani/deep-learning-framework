from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.variable import Variable

class Linear(Function):
    @staticmethod
    def run(variable: Variable) -> Variable:    
        return variable