import numpy as np

from dl.automatic_gradient.functions import Absolute

class L1:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, m, params):
        # params is the result of model.parameters() which is a List[Variable]: numpy of course (n, )
        # m is the number of examples in the batch for example
        return self._lambda / m * np.sum(Absolute.run(params))
    
    def __repr__(self):
        return f"L1(_lambda={self._lambda})"
