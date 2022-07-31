# Basic Deep Learning Framework <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Introduction](#introduction)
- [Installation](#installation)
  - [Usage](#usage)
- [Dependencies](#dependencies)
- [Elements](#elements)
  - [Layers](#layers)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
    - [Classification](#classification)
    - [Regression](#regression)
  - [Regularization Techniques](#regularization-techniques)
  - [Optimizers](#optimizers)
- [Code Examples](#code-examples)
  - [`import` Statements](#import-statements)
  - [Binary Classification Code Example](#binary-classification-code-example)
  - [Regression Code Example](#regression-code-example)
- [Implementation notes](#implementation-notes)

## Introduction

In this project, I introduce a fundamental deep learning framework for quick and easy prototyping, intended as a small project to deepen my understanding of deep learning concepts with syntax inspired by [TensorFlow](https://www.tensorflow.org/) and complete implementation from the Coursera [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) Course by professor [Andrew Ng](https://twitter.com/AndrewYNg)

## Installation

### Usage

Download this repository and copy the `dl` folder inside your working directory.

## Dependencies

- [Numpy](https://numpy.org/) for computations.
- [Dill](https://dill.readthedocs.io/en/latest/) for saving and loading deep learning models.

## Elements

### Layers

- Input
- Dense

### Activation Functions

- Rectified Linear Unit
- Leaky Rectified Linear Unit
- Tangent Hyperbolic
- Sigmoid
- Linear

### Loss Functions

#### Classification

- Binary Cross-Entropy
- Categorical Cross-Entropy

#### Regression

- Mean Squared Error
- Mean Absolute Error

### Regularization Techniques

- L1
- L2
- Dropout (Inverted Dropout)

### Optimizers

- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini Batch Gradient Descent
- Momentum with Gradient Descent
- RMSProp
- Adaptive Moment Estimation

## Code Examples

### `import` Statements

```python
from dl.model import Model
from dl.layers import Input, Dense
from dl.regularization import L1, L2, Dropout
from dl.functions.activation import ReLU, LeakyReLU, TanH, Sigmoid, Linear
# optimizers
from dl.optimizers import BatchGradientDescent
from dl.optimizers import StochasticGradientDescent
from dl.optimizers import MiniBatchGradientDescent
from dl.optimizers import MomentumGradientDescent
from dl.optimizers import RMSProp
from dl.optimizers import AdaptiveMomentEstimation
# loss functions
from dl.functions.loss import BinaryCrossEntropy
from dl.functions.loss import CategoricalCrossEntropy
from dl.functions.loss import MSE
from dl.functions.loss import MAE
```

### Binary Classification Code Example

```python
# dataset generation
n = 1 # number of features
m = 5000 # number of examples

X = np.random.randn(n, m)
y = (X[0, :] > 0.5).astype(int).reshape(-1, m)

# dataset split & shuffle into training and testing sets
m_train = np.ceil(0.8 * m).astype(int)

permutation = np.random.permutation(m)

X_train = X[:, permutation[:m_train]]
y_train = y[:, permutation[:m_train]]

X_test = X[:, permutation[m_train:]]
y_test = y[:, permutation[m_train:]]

# nerual network architecture definition
model = Model(layers=[
    Input(n_units=n),
    Dense(n_units=16, activation_function=ReLU(), dropout=Dropout(keep_prob=0.9)),
    Dense(n_units=16, activation_function=ReLU()),
    Dense(n_units=1, activation_function=Sigmoid())
])

model.summary()

# hyper-parameters definition
batch_size = 128
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
n_epochs = 300
_lambda = 10
decay_rate = 1e-3

# learning rate decay function
def lr_decay(lr0, epoch):
    return lr0

# optimizer definition
optimizer = BatchGradientDescent(learning_rate=learning_rate, lr_decay=lr_decay)
optimizer = StochasticGradientDescent(learning_rate=learning_rate, lr_decay=lr_decay)
optimizer = MiniBatchGradientDescent(batch_size=batch_size, learning_rate=learning_rate, lr_decay=lr_decay)
optimizer = MomentumGradientDescent(batch_size=batch_size, learning_rate=learning_rate, lr_decay=lr_decay, beta=beta1)
optimizer = RMSProp(batch_size=batch_size, learning_rate=learning_rate, lr_decay=lr_decay, beta=beta2)
optimizer = AdaptiveMomentEstimation(batch_size=batch_size, learning_rate=learning_rate, lr_decay=lr_decay, beta1=beta1, beta2=beta2)

# loss function definition
Loss = BinaryCrossEntropy(regularizer=L1(_lambda=_lambda))
Loss = BinaryCrossEntropy(regularizer=L2(_lambda=_lambda))
Loss = BinaryCrossEntropy()

model.compile(optimizer=optimizer, loss=Loss)

# history["loss"]: training loss
history = model.optimize(X=X_train, y=y_train, n_epochs=n_epochs)

# predict using the model
y_test_predictions = model(X_test)

# calculating the test loss
test_loss = Loss(y_test_predictions, y_test)

# calculating test accuracy
y_test_predictions = (y_test_predictions > 0.5).astype(int)

test_accuracy = np.mean(y_test_predictions == y_test)

print(f"test loss is {test_loss}")
print(f"test accuracy is {test_accuracy}")

# save the model
model.save(file_path)

# load the model
model = Model.load(file_path)

```

### Regression Code Example

```python
# dataset generation
n = 1 # number of features
m = 5000 # number of examples

X = np.random.randn(n, m)
y = 2 * X + 1

# dataset split & shuffle into training and testing sets
m_train = np.ceil(0.8 * m).astype(int)

permutation = np.random.permutation(m)

X_train = X[:, permutation[:m_train]]
y_train = y[:, permutation[:m_train]]

X_test = X[:, permutation[m_train:]]
y_test = y[:, permutation[m_train:]]


# nerual network architecture definition
model = Model(layers=[
    Input(n_units=n),
    Dense(n_units=16, activation_function=ReLU()),
    Dense(n_units=16, activation_function=TanH()),
    Dense(n_units=1, activation_function=Linear())
])

model.summary()

# hyper-parameters definition
batch_size = 128
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
n_epochs = 300
_lambda = 10
decay_rate = 1e-3

# learning rate decay function
def lr_decay(lr0, epoch):
    return lr0

# optimizer definition
optimizer = AdaptiveMomentEstimation(batch_size=batch_size, learning_rate=learning_rate, lr_decay=lr_decay, beta1=beta1, beta2=beta2)

# loss function definition
Loss = MSE()

model.compile(optimizer=optimizer, loss=Loss)

# history["loss"]: training loss
history = model.optimize(X=X_train, y=y_train, n_epochs=n_epochs)

# predict using the model
y_test_prediction = model(X_test)

# calculating the test loss
test_loss = Loss(y_test_prediction, y_test)
print(f"test loss is {test_loss}")

# save the model
model.save(file_path)

# load the model
model = Model.load(file_path)
```

## Implementation notes

- `Categorical Cross-Entropy` loss function implements a softmax activation and by adding a `Linear` activation function in the last layer your implementation would be correct.
- `X` must have the shape `(# of feature, # of examples)`.
