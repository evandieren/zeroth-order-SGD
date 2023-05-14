import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from difference_methods import one_point_estimate, two_point_estimate, coordinate_estimate

def forward_helper(x, w, b):
    o = x.mm(w.t())
    if b is not None:
        o += b.unsqueeze(0).expand_as(o)
    return o

class LinearFunction_Zeroth(Function):
    """ Version of affine function that uses zeroth order derivative estimation for gradient calculation """
    
    @staticmethod
    def forward(ctx, input, weight, bias=None, difference_method : str = None, mu : float = None, n : int = None):
        # Save differentiable values for backward
        ctx.save_for_backward(input, weight, bias)

        # Save non_differentiable values for backwards
        f = lambda x: forward_helper(x, w=weight, b=bias)
        if difference_method == "one":
            ctx.diff = lambda x: one_point_estimate(x, f, mu, n)
        elif difference_method == "two":
            ctx.diff = lambda x: two_point_estimate(x, f, mu, n)
        elif difference_method == "coord":
            ctx.diff = lambda x: coordinate_estimate(x, f, mu)
        else:
            raise ValueError(f"Input {difference_method=} must be 'one', 'two', or 'coord'.")
        ctx.mark_non_differentiable(difference_method, mu, n)
        
        # Return output
        return f(input)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack saved tensors
        input, weight, bias = ctx.saved_tensors
        # Initialise gradients to None
        grad_input = grad_weight = grad_bias = None
        
        # Calculate gradients if needed
        if ctx.needs_input_grad[0]:
            grad_input = ctx.diff(input) # include grad_output??
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias