# Numpy Deep Learning

Deep Learning Framework built entirely using numpy. The framework is built with the PyTorch design in mind and is 
meant to be used almost exactly like one would with with PyTorch. Implemented for the moment are fully connected
 layers, Tanh(), Sigmoid() and ReLU() activation layers, MSE and BCE loss and mini-batch Stochastic Gradient Descent 
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

## Use as package

Install the package from the root directory with:

```python
pip install -e .
```

You can then import it with :

```python
import numpy_dl
```

Usage examples are provided in the afore mentioned example_usage.py .

## Next improvements

Command line functions, optimizers and multi-threading/batch 
processing to improve speed.

