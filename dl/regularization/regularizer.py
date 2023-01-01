from dl.automatic_gradient.variable import Variable

class Regularizer:
    def __init__(self, _lambda: float):
        self._lambda = None

    def __call__(self, m: int, params) -> Variable:
        pass

    def __repr__(self):
        pass