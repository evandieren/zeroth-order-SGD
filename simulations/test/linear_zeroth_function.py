import math
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from difference_methods import one_point_estimate, two_point_estimate, coordinate_estimate

def forward_helper(x, w, b):
    o = x.mm(w.t())
    if b is not None:
        o += b.unsqueeze(0).expand_as(o)
    return o

class Linear_Zeroth_Function(Function):
    """ Version of affine function that uses zeroth order derivative estimation for gradient calculation """
    
    @staticmethod
    def forward(ctx, input, weight, bias=None, difference_method : str = None, mu : float = None, n : int = None):
        # Save differentiable values for backward
        ctx.save_for_backward(input, weight, bias)

        # Save difference method backwards
        if difference_method == "one":
            ctx.diff = lambda f, x: one_point_estimate(f, x, mu, n)
        elif difference_method == "two":
            ctx.diff = lambda f, x: two_point_estimate(f, x, mu, n)
        elif difference_method == "coord":
            ctx.diff = lambda f, x: coordinate_estimate(f, x, mu)
        else:
            raise ValueError(f"Input {difference_method=} must be 'one', 'two', or 'coord'.")
        
        # Return output
        return forward_helper(input, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack saved tensors
        input, weight, bias = ctx.saved_tensors
        # Initialise gradients to None
        grad_input = grad_weight = grad_bias = None
        
        # Calculate gradients if needed
        if ctx.needs_input_grad[0]:
            estimated_grad = ctx.diff(lambda x: forward_helper(input, x, bias), weight)
            grad_input = grad_output.mm(estimated_grad)
        if ctx.needs_input_grad[1]:
            estimated_grad = ctx.diff(lambda x: forward_helper(x, weight, bias), input)
            grad_weight = grad_output.t().mm(estimated_grad)
        if bias is not None and ctx.needs_input_grad[2]:
            estimated_grad = ctx.diff(lambda x: forward_helper(input, weight, x), bias)
            grad_bias = grad_output.mm(estimated_grad) # FIXME

        # Return gradients for inputs 
        return grad_input, grad_weight, grad_bias, None, None, None

class Linear_Zeroth(torch.nn.Module):
    """ Version of Linear module with zeroth order gradient back-end

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        difference_method: zeroth order method to use. Options are 'one', 'two', or 'coord'. 
            Default: ``None``.
        mu: norm of the difference step used in difference_method.
            Default: ``None``.
        n: number of estimates to average over in difference_method.
            Default: ``None``.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, difference_method : str = None, mu : float = None, n : int = None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.difference_method = difference_method
        self.mu = mu
        self.n = n

        self.weight = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return Linear_Zeroth_Function.apply(input, self.weight, self.bias, self.difference_method, self.mu, self.n)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, difference_method={}(mu={},n={})'.format(self.in_features, self.out_features, self.bias is not None, self.difference_method, self.mu, self.n)