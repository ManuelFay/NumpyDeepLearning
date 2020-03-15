# Numpy Deep Learning

Deep Learning Framework built entirely using numpy. The framework is built with the PyTorch design in mind and is 
meant to be used almost exactly like one would with with PyTorch. Implemented for the moment are fully connected
 layers, Tanh() and ReLU() activation layers, MSE loss and mini-batch Stochastic Gradient Descent 
 (Gradient Descent and Stochastic Gradient Descent are available upon choice of mini-batch size).
 
## Installation

Create a virtual python environment and source it.

```python
virtualenv venv -p python3.6
source venv/bin/activate
```

Install required packages (NumPy)

```python
pip install -r dev.requirements.txt
```

Run the example code.

```python
 python example_usage.py       
```

The example_usage.py is a usage example that generates synthetic binary data and then implements a multi-layer 
linear model to classify them. The framework however extends to any other usage and dataset.

## Next improvements

Command line functions, variable typing, working BCE implementation, optimizers and batch 
processing to improve speed.

