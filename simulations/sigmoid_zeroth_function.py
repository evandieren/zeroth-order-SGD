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
    def forward(ctx, input, difference_method : str = None, mu : float = None, n : int = None):
        # Save differentiable values for backward
        ctx.save_for_backward(input)

        # Save difference method backwards
        if difference_method == "one":
            ctx.diff = lambda x: one_point_estimate(forward_helper, x, mu, n)
        elif difference_method == "two":
            ctx.diff = lambda x: two_point_estimate(forward_helper, x, mu, n)
        elif difference_method == "coord":
            ctx.diff = lambda x: two_point_estimate(forward_helper, x, mu, n)
        else:
            raise ValueError(f"Input {difference_method=} must be 'one', 'two', or 'coord'.")
        # Mark non_differentiable inputs as such
        ctx.mark_non_differentiable(difference_method, mu, n)
        
        # Return output
        return forward_helper(input)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack saved tensors
        input = ctx.saved_tensors
        # Initialise gradients to None
        grad_input = None
        
        # Calculate gradients if needed
        if ctx.needs_input_grad[0]:
            estimated_grad = ctx.diff(input)
            grad_input = grad_output.mm(estimated_grad)

        # Return gradients for differentiable inputs 
        return grad_input