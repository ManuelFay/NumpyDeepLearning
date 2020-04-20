class Module:

    def __init__(self):
        self.is_training = True

    def __call__(self, *arg):
        return self.forward(*arg)

    def forward(self, *arg):
        """forward should get for input, and returns, a tensor or a tuple of tensors."""
        raise NotImplementedError

    def backward(self, grad):
        """backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect
        to the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input."""
        raise NotImplementedError

    def zero_grad(self):    # pylint: disable=no-self-use
        return None

    def param(self):
        """param should  return  a  list  of  pairs,  each  composed  of  a  parameter  tensor,  and  a  gradient
        tensor of same size.  This list should be empty for parameter less modules (e.g.  ReLU)"""
        return [x for x in vars(self).keys()]
