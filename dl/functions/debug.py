import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from dl.automatic_gradient import Variable
from dl.functions.loss import CategoricalCrossEntropy
from dl.functions.loss import BinaryCrossEntropy

def main():
    y = np.array([0, 1])
    y = Variable.from_numpy(y)

    y_pred = np.random.uniform(0, 1, size=(2, ))
    y_pred = Variable.from_numpy(y_pred)

    loss = BinaryCrossEntropy()
    print(loss(y_pred, y))

if __name__ == "__main__":
    main()