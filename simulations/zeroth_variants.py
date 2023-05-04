import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from difference_methods import one_point_estimate, two_point_estimate, coordinate_estimate

class LinearFunction_Zeroth(Function):
    """ Version of affine function that uses zeroth order derivative estimation for gradient calculation """
    

    @staticmethod
    def forward(ctx, input, weight, bias=None, difference_method : str = None, mu : float = None, n : int = None):
        # Save values for backward
        ctx.save_for_backward(input, weight, bias)
        if difference_method == "one":
            ctx.diff = lambda x, f: one_point_estimate(x, f, mu, n)
        elif difference_method == "two":
            ctx.diff = lambda x, f: two_point_estimate(x, f, mu, n)
        elif difference_method == "coord":
            ctx.diff = lambda x, f: coordinate_estimate(x, f, mu)
        else:
            raise ValueError(f"Input {difference_method=} must be 'one', 'two', or 'coord'.")
        ctx.mark_non_differentiable(difference_method, mu, n)
        # Calculate output
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack saved tensors
        input, weight, bias = ctx.saved_tensors
        # Initialise gradients to None
        grad_input = grad_weight = grad_bias = None

        # Setup function for difference methods
        def f(x=input, w=weight, b=bias):
            o = x.mm(w.t())
            if b is not None:
                o += b.unsqueeze(0).expand_as(o)
            return o
        
        # Calculate gradients if needed
        if ctx.needs_input_grad[0]:
            my_f = lambda x: f(x=x)
            grad_input = ctx.diff(input, my_f) # include grad_output??
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
