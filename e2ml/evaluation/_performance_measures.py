import numpy as np

from sklearn.utils.validation import check_consistent_length, check_scalar, column_or_1d

from . import zero_one_loss


def confusion_matrix(y_true, y_pred, *, n_classes=None, normalize=None):
    """Compute confusion matrix to evaluate the accuracy of a classifier.

    By definition a confusion matrix `C` is such that `C_ij` is equal to the number of observations known to be class
    `i` and predicted to be in class `j`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Expected to be in the set `{0, ..., n_classes-1}`.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier. Expected to be in the set `{0, ..., n_classes-1}`.
    n_classes : int
        Number of classes. If `n_classes=None`, the number of classes is assumed to be the maximum value of `y_ture`
        and `y_pred`.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None,
        confusion matrix will not be normalized.

    Returns
    -------
    C : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the number of amples with true label being
        i-th class and predicted label being j-th class.
    """
    y_true = column_or_1d(y_true, dtype=int)
    y_pred = column_or_1d(y_pred, dtype=int)

    check_consistent_length(y_true, y_pred)

    y_min = np.min((y_true.min(), y_pred.min()))
    y_max = np.max((y_true.max(), y_pred.max()))

    if y_min < 0:
        raise ValueError("`y_true` and `y_pred` are expected to have values in the set `{0, ..., n_classes-1}`.")

    if n_classes is None:
        n_classes = int(y_max+1)

    check_scalar(n_classes, name="n_classes", min_val=1, target_type=int)

    C = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            C[i, j] = np.sum((y_true == i) & (y_pred == j))

    with np.errstate(all='ignore'):
        if normalize == 'true':
            C = C/C.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            C = C/C.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            C = C/C.sum()

        C = np.nan_to_num(C)

    return C


def accuracy(y_true, y_pred):
    """Computes the accuracy of the predicted class label `y_pred` regarding the true class labels `y_true`.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True class labels as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted class labels as array-like object.

    Returns
    -------
    acc : float in [0, 1]
        Accuracy.
    """
    return 1 - zero_one_loss(y_true=y_true, y_pred=y_pred)


def cohen_kappa(y_true, y_pred, n_classes=None):
    """Compute Cohen's kappa: a statistic that measures agreement between true and predicted class labeles.

    This function computes Cohen's kappa, a score that expresses the level of agreement between true and predicted class
    labels. It is defined as

    kappa = (P_o - P_e) / (1 - P_e),

    where `P_o` is the empirical probability of agreement on the label assigned to any sample (the observed agreement
    ratio), and `P_e` is the expected agreement when true and predicted class labels are assigned randomly.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Expected to be in the set `{0, ..., n_classes-1}`.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier. Expected to be in the set `{0, ..., n_classes-1}`.
    n_classes : int
        Number of classes. If `n_classes=None`, the number of classes is assumed to be the maximum value of `y_ture`
        and `y_pred`.

    Returns
    -------
    kappa : float in [-1, 1]
        The kappa statistic between -1 and 1.
    """
    C = confusion_matrix(y_true=y_true, y_pred=y_pred, n_classes=n_classes)
    n_classes = len(C)
    c0 = np.sum(C, axis=0)
    c1 = np.sum(C, axis=1)
    expected = np.outer(c0, c1) / np.sum(c0)
    w_mat = np.ones((n_classes, n_classes), dtype=int)
    w_mat.flat[:: n_classes + 1] = 0
    kappa = 1 - np.sum(w_mat * C) / np.sum(w_mat * expected)
    return kappa


def macro_f1_measure(y_true, y_pred, n_classes=None):
    """Computes the macro F1 measure.

    The F1 measure is computed for each class individually and then averaged. If there is a class label with no true nor
    predicted samples, the F1 measure is set to 0.0 for this class label.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Expected to be in the set `{0, ..., n_classes-1}`.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier. Expected to be in the set `{0, ..., n_classes-1}`.
    n_classes : int
        Number of classes. If `n_classes=None`, the number of classes is assumed to be the maximum value of `y_true`
        and `y_pred`.

    Returns
    -------
    macro_f1 : float in [0, 1]
        The macro f1 measure between 0 and 1.
    """
    C = confusion_matrix(y_true=y_true, y_pred=y_pred, n_classes=n_classes)
    n_classes = len(C)
    macro_f1 = 0
    for c in range(n_classes):
        with np.errstate(all="ignore"):
            tp_c = C[c, c]
            fp_c = C[:, c].sum() - C[c, c]
            fn_c = C[c, :].sum() - C[c, c]
            prec_c = tp_c / (tp_c + fp_c)
            rec_c = tp_c / (tp_c + fn_c)
            f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c)
            macro_f1 += np.nan_to_num(f1_c)
    macro_f1 /= n_classes
    return macro_f1