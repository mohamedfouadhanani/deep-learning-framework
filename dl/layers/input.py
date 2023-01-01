from dl.layers.layer import Layer

class Input(Layer):
    def __init__(self, n_units: int):
        self.n_units = n_units

    def __repr__(self):
        return f"Input(n_units={self.n_units})"
