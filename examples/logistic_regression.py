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

inputs = np.random.randn(500, 3)
outputs = (inputs.sum(axis=1, keepdims=True) > 0).astype(int)

model = Model([
    Dense(3, 8),
    TanH(),
    Dense(8, 16),
    ReLU(),
    # Dropout(keep_prob=0.8),
    Dense(16, 1),
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