import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from dl.automatic_gradient import Variable
from dl.functions.loss import CategoricalCrossEntropy

def main():
    y = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    y = Variable.from_numpy(y)

    y_pred = np.random.uniform(0, 1, size=(2, 3))

    y_pred = Variable.from_numpy(y_pred)

    loss = CategoricalCrossEntropy()
    print(loss(y_pred, y))

if __name__ == "__main__":
    main()