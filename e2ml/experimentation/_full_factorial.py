"""
This code implement a full factorial approach.
"""

import numpy as np


def full_fac(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    X : np.ndarray of shape (n_combs, n_factors)
        The design matrix with coded levels 0 to k-1 for a k-level factor

    Example
    -------
    ::

        >>> full_fac([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
    """
    levels = np.array(levels)

    n_factors = len(levels)
    n_combs = np.prod(levels)
    X = np.zeros((n_combs, n_factors))
    
    fact_levs = np.zeros(n_combs)
    for i in range(n_factors):
        n_repeats = np.prod(levels[:i])
        for j in range(n_combs):
            fact_levs[j] = (j // n_repeats) % levels[i]

        X[:, i] = fact_levs

    return X

    
