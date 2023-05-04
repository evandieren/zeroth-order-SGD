import numpy as np

def n_sphere_sample(shape):
    """ Rejection method for sampling on unit-sphere """
    sample = np.array([2])
    while np.linalg.norm(sample, ord=2) > 1:
        sample = np.random.uniform(-1.0, 1.0, shape)
    return sample / np.linalg.norm(sample, ord=2)

def one_point_estimate(f, x : np.ndarray, mu : float, n : int = 1):
    """ Single point estimate for derivative """
    # Checks
    assert n > 0, f"Input {n=} must be greater than zero."
    # Calculate output
    output = np.zeros(x.shape)
    for _ in range(0,n):
        # Sample
        u_i = n_sphere_sample(x.shape)
        output += f(x + mu * u_i) * u_i
    # Return
    return (x.size/mu) * output

def two_point_estimate(f, x : np.ndarray, mu : float, n : int = 1):
    """ Two point estimate for derivative """
    # Checks
    assert n > 0, f"Input {n=} must be greater than zero."
    # Calculate output
    output = np.zeros(x.shape)
    for _ in range(0,n):
        # Sample
        u_i = n_sphere_sample(x.shape)
        # Calculate next term
        output += (f(x + mu * u_i) - f(x)) * u_i
    # Return
    return (x.size/mu) * output

def coordinate_estimate(f, x : np.ndarray, mu : float):
    """ Two point estimate for derivative """
    # Calculate output
    output = np.zeros(x.shape)
    for i in range(0,x.size):
        # Construction elementary vector
        e_i = np.zeros(x.shape).flatten()
        e_i[i] = 1
        e_i = e_i.reshape(x.shape)
        # Calculate next term
        output += (f(x + mu * e_i) - f(x)) * e_i
    # Return
    return (1/mu) * output