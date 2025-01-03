import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z
    NumPy has a function called exp(), which offers a convenient way to calculate the exponential
    (  𝑒^𝑧  ) of all elements in the input array (z).

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1 / ( 1 + np.exp(-z) )

    return g