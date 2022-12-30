import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from decimal import Decimal

sys.path.append(os.getcwd())

from dl.automatic_gradient import Variable
from dl.automatic_gradient.functions import TanH, ReLU, Exponential, Softmax

def weights(n_inputs, n_outputs):
    raw_W = np.random.randn(n_outputs, n_inputs)
    np.random.randn(n_outputs, n_inputs) * np.sqrt(1 / n_inputs)
    h, w = raw_W.shape

    W = []
    for i in range(h):
        w_i = []
        for j in range(w):
            w_i.append(Variable(data=raw_W[i, j]))
        
        W.append(w_i)
    
    W = np.array(W)
    return W

def numpy_operations():
    m = 5
    n = 1
    t = 2
    
    W = weights(n, t)
    print("W")
    print(W)

    print()

    X = weights(n, m)
    print("X")
    print(X)

    print()

    b = weights(t, 1)
    print("b")
    print(b)

    print()

    print("X • W.T + b")
    print(np.dot(X, W.T) + b)

def zero_gradients(W):
    for w in W:
        for param in w.reshape(-1, ):
            param.gradient = Decimal(0.)
    
def basic_regression_nn():
    m = 5
    n = 1

    X = np.random.randn(m, n)
    y = 2 * X + 1

    # 1 3 1

    W1 = weights(1, 3)
    b1 = weights(3, 1)

    # print(W1.shape)
    # print(b1.shape)

    W2 = weights(3, 1)
    b2 = weights(1, 1)

    # print(W2.shape)
    # print(b2.shape)

    alpha0 = Decimal(0.001)
    iterations = 10000
    losses = []
    alphas = []
    for iteration in range(iterations):
        # alpha = 0.95 ** iteration * alpha0
        alpha = alpha0
        alphas.append(alpha)

        Z1 = np.dot(X, W1.T) + b1
        A1 = TanH.run(Z1)

        Z2 = np.dot(A1, W2.T) + b2
        A2 = Z2

        loss = ((y - A2) ** 2).sum() / len(X)
        losses.append(loss.data)

        zero_gradients((W1, b1, W2, b2))
        
        loss.backward()
        for w in (W1, b1, W2, b2):
            for param in w.reshape(-1, ):
                param.data -= alpha * param.gradient
        
        if iteration % 500 == 0:
            print(f"iteration {iteration}:")
            print(f"\tloss: {loss.data}")

    print(f"loss: {loss.data}")
    figure, axis = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    figure.tight_layout(pad=5)

    axis[0].plot(list(range(iterations)), losses)
    axis[0].set_xlabel("iteration")
    axis[0].set_ylabel("loss")
    axis[0].set_title("Loss with respect to iterations")

    axis[1].plot(list(range(iterations)), alphas)
    axis[1].set_xlabel("iteration")
    axis[1].set_ylabel("alpha")
    axis[1].set_title("Alpha decay with respect to iterations")

    plt.show()

def linear_regression():
    m = 5
    n = 1

    X = np.random.randn(m, n)
    y = 2 * X + 1

    w = weights(1, 1)
    b = weights(1, 1)

    alpha = Decimal(0.001)
    iterations = 10000
    losses = []
    for iteration in range(iterations):
        y_hat = np.dot(X, w.T) + b

        loss = ((y - y_hat) ** 2).sum() / len(X)
        losses.append(loss.data)

        zero_gradients((w, b))
        
        loss.backward()
        for w_i in (w, b):
            for param in w_i.reshape(-1, ):
                param.data -= alpha * param.gradient
        
        if iteration % 500 == 0:
            print(f"iteration {iteration}:")
            print(f"\tloss: {loss.data}")
    
    print(f"loss: {loss.data}")
    figure, axis = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    figure.tight_layout(pad=5)

    axis[0].plot(list(range(iterations)), losses)
    axis[0].set_xlabel("iteration")
    axis[0].set_ylabel("loss")
    axis[0].set_title("Loss with respect to iterations")

    # axis[1].plot(list(range(iterations)), alphas)
    # axis[1].set_xlabel("iteration")
    # axis[1].set_ylabel("alpha")
    # axis[1].set_title("Alpha decay with respect to iterations")

    plt.show()

def test_softmax():
    x1 = Variable(1.0, label="x1")
    x2 = Variable(2.0, label="x2")
    vector_1 = np.array([x1, x2])

    y1 = Variable(3.0, label="y1")
    y2 = Variable(5.0, label="y2")
    vector_2 = np.array([y1, y2])

    z1 = Variable(2.0, label="z1")
    z2 = Variable(5.0, label="z2")
    vector_3 = np.array([z1, z2])

    vector = np.array([vector_1, vector_2, vector_3])

    print("vector")
    print(vector.shape)

    output = Softmax.run(vector, axis=1)

    print("output")
    print(output.shape)

def from_numpy_test():
    ndarray = Variable.from_numpy(np.random.randn(3, 2))
    print(ndarray.shape)
    print(type(ndarray[0, 0]))

if __name__ == "__main__":
    # basic_regression_nn()
    # linear_regression()
    # test_softmax()
    from_numpy_test()