import numpy as np

from dl.automatic_gradient.functions.function import Function
from dl.automatic_gradient.functions.exponential import Exponential
from dl.automatic_gradient.variable import Variable

class Softmax(Function):
    @staticmethod
    def run(variables, axis=None):
        def atomic_softmax(vector):
            # atomic_softmax was created to be able to use numpys np.apply_along_axis function
            exps = Exponential.run(vector)
            summation = exps.sum()
            atomic_result = exps / summation
            
            return atomic_result
        
        results = np.apply_along_axis(atomic_softmax, axis, variables)

        return results

Softmax.run = np.vectorize(Softmax.run, excluded=[0])