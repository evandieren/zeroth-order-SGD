import numpy as np
import torch

def n_sphere_sample(shape):
    """ Rejection method for sampling on unit-sphere """
    sample = np.random.normal(loc=0, scale=1, size=shape)
    return np.float32((sample / np.linalg.norm(sample, ord=2)))

def one_point_estimate(f, x : np.ndarray, mu : float, n : int = 1):
    """ Single point estimate for derivative """
    assert n > 0, f"Input {n=} must be greater than zero."
    # Calculate output
    output = torch.zeros_like(x)
    for _ in range(0,n):
        # Sample
        u_i = n_sphere_sample(x.shape)
        output += f(x + mu * u_i).sum() * u_i
    # Return
    return (np.prod(x.size())/(n*mu)) * output

def two_point_estimate(f, x : np.ndarray, mu : float, n : int = 1):
    """ Two point estimate for derivative """
    # Checks
    assert n > 0, f"Input {n=} must be greater than zero."
    # Calculate output
    output = torch.zeros_like(x)
    for _ in range(0,n):
        # Sample
        u_i = n_sphere_sample(x.shape)
        # Calculate next term
        output += (f(x + mu * u_i) - f(x)).sum() * u_i
    # Return
    return (np.prod(x.size())/(n*mu)) * output

def coordinate_estimate(f, x : np.ndarray, mu : float):
    """ Coordinate estimate for derivative """
    # Calculate output
    output = torch.zeros_like(x)
    for i in range(0,np.prod(output.size())):
        # Construction elementary vector
        e_i = torch.zeros_like(x).flatten()
        e_i[i] = 1
        e_i = e_i.reshape(x.shape)
        # Calculate next term
        output += (f(x + mu * e_i) - f(x)).sum() * e_i
    # Return
    return (1/mu) * output