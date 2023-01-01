import numpy as np
from typing import List

from dl.automatic_gradient.variable import Variable
from dl.regularization.regularizer import Regularizer

class L2(Regularizer):
    def __init__(self, _lambda: float):
        self._lambda = _lambda

    def __call__(self, m: int, params) -> Variable:
        # params is the result of model.parameters() which is a List[Variable]: numpy of course (n, )
        # m is the number of examples in the batch for example

        return self._lambda / (2 * m) * np.dot(params, params.T)

    def __repr__(self):
        return f"L2(_lambda={self._lambda})"