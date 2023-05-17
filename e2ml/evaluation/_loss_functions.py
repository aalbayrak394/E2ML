import numpy as np

from sklearn.utils import column_or_1d
from scipy.special import xlogy


def zero_one_loss(y_true, y_pred):
    """
    Computes the empirical risk for the zero-one loss function.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True class labels as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted class labels as array-like object.

    Returns
    -------
    risk : float in [0, 1]
        Empirical risk computed via the zero-one loss function.
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    # Compute and return the empirical risk.
    loss = y_true != y_pred

    n = len(y_true)
    empirical_risk = 1/n * sum(loss)

    return empirical_risk


def binary_cross_entropy_loss(y_true, y_pred):
    """
    Computes the empirical risk for the binary cross entropy (BCE) loss function.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True conditional class probabilities as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted conditional class probabilities as array-like object.

    Returns
    -------
    risk : float in [0, +infinity]
        Empirical risk computed via the BCE loss function.
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    # Check value ranges of probabilities and raise ValueError if the ranges are invalid. In this case, it should be
    # allowed to have estimated probabilities in the interval [0, 1] instead of only (0, 1).
    if np.any(y_pred < 0) or np.any(y_pred > 1):
        raise ValueError

    # Compute and return the empirical risk.
    loss = xlogy(-y_true, y_pred) - xlogy(1-y_true, 1-y_pred)

    n = len(y_true)
    empirical_risk = 1/n * sum(loss)

    return empirical_risk
