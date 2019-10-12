# Numpy Deep Learning

Deep Learning Framework built entirely using numpy. The framework is built with the PyTorch design in mind and is meant to be used almost exactly like one would with with PyTorch. 

Implemented for the moment are fully connected layers, Tanh() and ReLU() activation layers, MSE loss and mini-batch Stochastic Gradient Descent (Gradient Descent and Stochastic Gradient Descent are available upon choice of mini-batch size).

Future improvements will include different optimizers, the addition of a CrossEntropy loss criterion, convolutional layers and speed improvements to parallelize the forward pass and a parser to add command lines arguments.

The test_npy.py is a usage example that generates binary data and then implements a multi-layer linear model to classify them. The framework however extends to any other usage and dataset.

