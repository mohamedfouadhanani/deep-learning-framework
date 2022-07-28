import numpy as np

class Dropout:
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    
    def __call__(self, A):
        d = np.random.randn(*A.shape)
        d = (d > self.keep_prob).astype(int)
        A = d * A
        A /= self.keep_prob

        return A