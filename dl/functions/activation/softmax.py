import numpy as np
from typing import List

from dl.automatic_gradient.variable import Variable
from dl.functions.activation.activation_function import ActivationFunction
from dl.automatic_gradient.functions.exponential import Exponential

class Softmax(ActivationFunction):
    @staticmethod
    def run(variables: List[Variable], axis=None) -> List[Variable]:
        def atomic_softmax(vector):
            # atomic_softmax was created to be able to use numpys np.apply_along_axis function
            exps = Exponential.run(vector)
            summation = exps.sum()
            atomic_result = exps / summation
            
            return atomic_result
        
        results = np.apply_along_axis(atomic_softmax, axis, variables)

        return results

Softmax.run = np.vectorize(Softmax.run, excluded=[0])