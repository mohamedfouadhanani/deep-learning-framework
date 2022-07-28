class Input:
    def __init__(self, n_units):
        self.n_units = n_units

    def __repr__(self):
        return f"\tInput_Layer:\n\t\toutput shape: ({self.n_units}, 1)"
