import numpy as np

def train_test_split(inputs, labels, percentage):
    m, _ = inputs.shape
    permutation = np.random.permutation(m)
    m_train = np.ceil(m * percentage).astype(int)

    training_inputs = inputs[permutation[:m_train]]
    testing_inputs = inputs[permutation[m_train:]]

    training_labels = labels[permutation[:m_train]]
    testing_labels = labels[permutation[m_train:]]

    return training_inputs, training_labels, testing_inputs, testing_labels