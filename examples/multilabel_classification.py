import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import pyplot as plt

from dl import Model
from dl import Dense
from dl.activations import TanH, ReLU, Sigmoid
from dl.losses import BinaryCrossEntropy
from dl.regularization import Dropout
from dl.optimizers import MiniBatchGradientDescent

m, n = 1000, 4
inputs = np.random.uniform(-1, 1, size=(m, n))
outputs = (inputs > 0).astype(np.int64)

model = Model([
    Dense(n, 8),
    ReLU(),
    Dense(8, 16),
    ReLU(),
    Dense(16, 32),
    TanH(),
    # Dropout(keep_prob=0.8),
    Dense(32, n),
    Sigmoid()
])

loss = BinaryCrossEntropy()

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