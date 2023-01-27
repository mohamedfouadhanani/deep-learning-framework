import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import pyplot as plt

from dl import Model
from dl import Dense
from dl.activations import TanH, ReLU
from dl.losses import CategoricalCrossEntropy
from dl.regularization import Dropout
from dl.optimizers import MiniBatchGradientDescent

m, n = 1000, 4
inputs = np.random.uniform(-1, 1, size=(m, n))
y = (inputs > 0).sum(axis=1)

outputs = np.zeros((m, n + 1))
outputs[range(m), y] = 1

model = Model([
    Dense(n, 8),
    ReLU(),
    Dense(8, 16),
    ReLU(),
    Dense(16, 32),
    TanH(),
    # Dropout(keep_prob=0.8),
    Dense(32, n + 1)
])

loss = CategoricalCrossEntropy()

n_epochs = 100
learning_rate = 0.01
batch_size = 128

optimizer = MiniBatchGradientDescent(batch_size, learning_rate)

model.compile(loss, optimizer)

history = model.optimize(inputs, outputs, n_epochs)

plt.plot(list(range(len(history["losses"]))), history["losses"])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()