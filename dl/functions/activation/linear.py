from dl.automatic_gradient.variable import Variable
from dl.functions.activation.activation_function import ActivationFunction

class Linear(ActivationFunction):
    @staticmethod
    def run(variable: Variable) -> Variable:
        return variable