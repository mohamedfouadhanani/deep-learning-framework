# Basic Deep Learning Framework <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Components](#components)
  - [Layers](#layers)
  - [Normalization](#normalization)
  - [Weights Initializers](#weights-initializers)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
    - [Classification](#classification)
    - [Regression](#regression)
  - [Regularization Techniques](#regularization-techniques)
  - [Optimizers](#optimizers)
- [Code Examples](#code-examples)
- [Implementation notes](#implementation-notes)

## Introduction

A from-scratch basic deep learning framework implementation in Python with NumPy, with syntax similar to TensorFlow and implementation similar to the lectures given by professor [Andrew Ng](https://www.andrewng.org/) in the course [The Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/?utm_medium=referral&utm_source=andrew-website).

## Dependencies

- [Numpy](https://numpy.org/) for computations.
- [Dill](https://dill.readthedocs.io/en/latest/) for saving and loading deep learning models.

## Components

### Layers

- Dense

### Normalization

- Batch Normalization
- Layer Normalization

### Weights Initializers

- Random Normal
- Random Uniform
- He Normal
- He Uniform
- Xavier Normal
- Xavier Uniform

### Activation Functions

- Rectified Linear Unit
- Leaky Rectified Linear Unit
- Tangent Hyperbolic
- Sigmoid
- Exponential Linear Unit
- Softmax

### Loss Functions

#### Classification

- Binary Cross-Entropy
- Categorical Cross-Entropy

#### Regression

- Mean Squared Error
- Mean Absolute Error

### Regularization Techniques

- Dropout (Inverted Dropout)

### Optimizers

- Stochastic Gradient Descent
- Momentum with Gradient Descent
- RMSProp
- Adaptive Moment Estimation

## Code Examples

Checkout the code examples in the `examples` directory.

## Implementation notes

- `inputs` must have the shape `(# of samples, # of features)`.
