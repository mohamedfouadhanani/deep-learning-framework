import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import pyplot as plt

from dl import Model
from dl import Dense
from dl.activations import TanH, ReLU
from dl.losses import MAE, MSE
from dl.regularization import Dropout
from dl.optimizers import BatchGradientDescent
from dl.optimizers import StochasticGradientDescent
from dl.optimizers import MiniBatchGradientDescent

m, n = 1000, 1
inputs = np.random.randn(m, n)
outputs = 2 * inputs + 1

model = Model([
    Dense(n, 8),
    TanH(),
    Dense(8, 16),
    ReLU(),
    Dense(16, 1)
])

n_epochs = 100
learning_rate = 0.01
batch_size = 128

loss = MSE()
# loss = MAE()

# optimizer = BatchGradientDescent(learning_rate)
# optimizer = StochasticGradientDescent(learning_rate)
optimizer = MiniBatchGradientDescent(batch_size, learning_rate)

model.compile(loss, optimizer)

history = model.optimize(inputs, outputs, n_epochs)

plt.plot(list(range(len(history["losses"]))), history["losses"])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()