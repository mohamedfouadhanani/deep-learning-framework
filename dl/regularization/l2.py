import numpy as np


class L2:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, m, params):
        # params is the result of model.parameters() which is a List[Variable]: numpy of course (n, )
        # m is the number of examples in the batch for example

        return self._lambda / (2 * m) * np.dot(params, params.T)

    def __repr__(self):
        return f"L2(_lambda={self._lambda})"