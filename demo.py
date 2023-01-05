from matplotlib import pyplot as plt
import numpy as np
from dl.model import Model
from dl.layers import Input, Dense
from dl.functions.activation import Linear, ReLU, TanH
from dl.functions.loss import MSE
from dl.optimizers import AdaptiveMomentEstimation

n = 3 # number of features
m = 200 # number of examples

X = np.random.randn(m, n)
y = 2 * X + 1

# dataset split & shuffle into training and testing sets
m_train = np.ceil(0.8 * m).astype(int)

permutation = np.random.permutation(m)

X_train = X[permutation[:m_train], :]
y_train = y[permutation[:m_train], :]

X_test = X[permutation[m_train:], :]
y_test = y[permutation[m_train:], :]

# nerual network architecture definition
model = Model(layers=[
    Input(n_units=n),
    Dense(n_units=4, activation_function=ReLU),
    Dense(n_units=3, activation_function=TanH),
    Dense(n_units=1, activation_function=Linear)
])

# hyper-parameters definition
batch_size = 256
learning_rate = 0.1
n_epochs = 100
_lambda = 10
decay_rate = 1e-3
beta1 = 0.9
beta2 = 0.999

# learning rate decay function
def lr_decay(lr0, epoch):
    return lr0

# optimizer definition
optimizer = AdaptiveMomentEstimation(batch_size=batch_size, learning_rate=learning_rate, lr_decay=lr_decay, beta1=beta1, beta2=beta2)

# loss function definition
Loss = MSE()

model.compile(optimizer=optimizer, loss=Loss)

print(model)
input("...")

history = model.optimize(X=X_train, y=y_train, n_epochs=n_epochs, verbose=True)

# predict using the model
y_test_prediction = model(X_test)

# calculating the test loss
test_loss = Loss(y_test_prediction, y_test)
print(f"test loss is {test_loss.data}")

losses = history["loss"]

plt.plot(list(range(len(losses))), losses)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("MSE loss w.r.t. iteration")
plt.show()