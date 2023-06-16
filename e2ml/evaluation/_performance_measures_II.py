import numpy as np
import matplotlib.pyplot as plt

def roc_curve(labels, x, scores):
    """
    Generate the Receiver Operating Characteristic (ROC) curve for a binary classification problem.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        True class labels for each sample.
    x : int or str
        Positive class or class of interest.
    scores : array-like of shape (n_samples,)
        Scores or probabilities assigned to each sample.

    Returns
    -------
    roc_curve : ndarray of shape (n_thresholds, 2)
        Array containing the true positive rate (TPR) and false positive rate (FPR) pairs
        at different classification thresholds.

    """
    # TODO


def roc_auc(points):
    """
    Compute the Area Under the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    points : array-like
        List of points representing the ROC curve.

    Returns
    -------
    auc : float
        Area Under the ROC curve.

    """
    # TODO 



def draw_lift_chart(true_labels, pos, predicted):
    """
    Draw a Lift Chart based on the true labels, positive class, and predicted class labels.

    Parameters
    ----------
    true_labels : array-like
        True class labels for each sample.
    pos : int or str
        Positive class or class of interest.
    predicted : array-like
        Predicted class labels for each sample.

    Returns
    -------
    None

    """
    plt.figure(figsize=(3,3))
    plt.xlabel('Dataset Size')
    plt.ylabel('True Positives')

    tp = (predicted == true_labels) & (predicted == pos)
    x = np.arange(len(predicted))
    y = np.cumsum(tp)

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()


def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions `p` and `q`.

    Parameters
    ----------
    p : array-like
        Probability distribution P.
    q : array-like
        Probability distribution Q.

    Returns
    -------
    kl_div : float
        KL divergence between P and Q.

    """
    p = np.array(p)
    q = np.array(q)

    log_ratio = np.log(p/q)
    kl_div = np.sum(p * log_ratio)

    return kl_div
