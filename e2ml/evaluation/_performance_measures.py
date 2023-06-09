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
    y_true = column_or_1d(y_true).astype(int)
    y_pred = column_or_1d(y_pred).astype(int)

    check_consistent_length(y_true, y_pred)
    # TODO
    # check_scalar(y, target_type=int, name='y')
    # check_scalar(y, target_type=int, name='y')

    y_min = np.min((y_true.min(), y_pred.min()))
    y_max = np.max((y_true.max(), y_pred.max()))

    if y_min < 0:
        raise ValueError('negative value')

    if n_classes is None:
        n_classes = int(y_max+1)

    C = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            C[i, j] = np.sum((y_true == i) & (y_pred == j))

    with np.errstate(all='ignore'):
        if normalize == 'true':
            C = C/C.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            C = C/C.sum(axis=1, keepdims=True)
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
    y_true = column_or_1d(y_true).astype(int)
    y_pred = column_or_1d(y_pred).astype(int)
    check_consistent_length(y_true, y_pred)

    C = confusion_matrix(y_true, y_pred)
    TN = C[0,0]
    TP = C[1,1]

    acc = np.trace(C) / np.sum(C)
    return acc


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
    y_true = column_or_1d(y_true).astype(int)
    y_pred = column_or_1d(y_pred).astype(int)
    check_consistent_length(y_true, y_pred)


    y_min = np.min((y_true.min(), y_pred.min()))
    y_max = np.max((y_true.max(), y_pred.max()))

    if y_min < 0:
        raise ValueError('negative value')

    if n_classes is None:
        n_classes = int(y_max+1)

    # C = confusion_matrix(y_pred=y_pred, y_true=y_true)
    # f1_scores = np.zeros(n_classes)

    # for c in range(n_classes):
    #     if not np.any(np.isnan(y_true)) and not np.any(np.isnan(y_pred)):
    #         f1 = 0.0
    #     else:
    #         TP = C[c, c]
    #         FP = np.sum(C[:, c]) - TP
    #         FN = np.sum(C[c]) - TP

    #         Prec = TP / (TP + FP)
    #         Rec = TP / (TP + FN)
    #         f1 = (2*Prec*Rec) / (Prec + Rec)

    #     f1_scores[c] = f1

    # return np.mean(f1_scores)

    C = confusion_matrix(y_true=y_true, y_pred=y_pred, n_classes=n_classes)
    n_classes = len(C)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        if C[c, :].sum() == 0 and C[:, c].sum() == 0:
            f1[c] = 0.0
        else:
            f1[c] = 2 * C[c, c] / (C[c, :].sum() + C[:, c].sum())
    return np.mean(f1)