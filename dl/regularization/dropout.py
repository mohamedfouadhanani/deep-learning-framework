import numpy as np

class Dropout:
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    
    def __call__(self, A, **kwargs):
        is_optimizing = kwargs["is_optimizing"]

        if not is_optimizing:
            return A
        
        d = np.random.uniform(0, 1, size=A.shape)
        d = (d < self.keep_prob).astype(int)

        A = A * d
        A /= self.keep_prob

        return A
    
    def initialize(self):
        pass
    
    def __repr__(self):
        return f"Dropout(keep_prob={self.keep_prob})"