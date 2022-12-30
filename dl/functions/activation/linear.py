from dl.functions.activation.activation_function import ActivationFunction

class Linear(ActivationFunction):
    @staticmethod
    def run(variable):
        return variable