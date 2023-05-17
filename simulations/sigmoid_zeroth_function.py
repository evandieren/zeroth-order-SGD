import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from difference_methods import one_point_estimate, two_point_estimate, coordinate_estimate

def forward_helper(x):
    return torch.sigmoid(x)

class Sigmoid_Zeroth_Function(Function):
    """ Version of affine function that uses zeroth order derivative estimation for gradient calculation """
    
    @staticmethod
    def forward(ctx, input, difference_method : int = None, mu : float = None, n : int = None):
        # Save differentiable values for backward
        ctx.save_for_backward(input)

        # Save difference method backwards
        if difference_method == "one":
            ctx.diff = lambda x: one_point_estimate(forward_helper, x, mu, n)
        elif difference_method == "two":
            ctx.diff = lambda x: two_point_estimate(forward_helper, x, mu, n)
        elif difference_method == "coord":
            ctx.diff = lambda x: coordinate_estimate(forward_helper, x, mu)
        else:
            raise ValueError(f"Input {difference_method=} must be 'one', 'two', or 'coord'.")
        
        # Return output
        return forward_helper(input)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack saved tensors
        input, = ctx.saved_tensors
        # Initialise gradients to None
        grad_input = None
        
        # Calculate gradients if needed
        if ctx.needs_input_grad[0]:
            estimated_grad = ctx.diff(input)
            grad_input = grad_output * estimated_grad # FIXME

        # Return gradients for inputs 
        return grad_input, None, None, None

class Sigmoid_Zeroth(torch.nn.Module):
    """ Version of ReLU module with zeroth order gradient back-end

    Args:
        difference_method: zeroth order method to use. Options are 'one', 'two', or 'coord'. 
            Default: ``None``.
        mu: norm of the difference step used in difference_method.
            Default: ``None``.
        n: number of estimates to average over in difference_method.
            Default: ``None``.
    """

    def __init__(self, difference_method : int = None, mu : float = None, n : int = None) -> None:
        super().__init__()

        self.difference_method = difference_method
        self.mu = mu
        self.n = n

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return Sigmoid_Zeroth_Function.apply(input, self.difference_method, self.mu, self.n)

    def extra_repr(self) -> str:
        return 'difference_method={}(mu={},n={})'.format(self.difference_method, self.mu, self.n)