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
    # Convert labels and scores to numpy arrays
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Get scores for true and false positives into 2 arrays and their total counts
    tp, fp = scores[labels == x], scores[labels != x]
    tpN, fpN = len(tp), len(fp)
    
    # Generate arrays for counting
    tmp = np.hstack([[tp, np.zeros(tpN), np.ones(tpN)], [fp, np.ones(fpN), np.zeros(fpN)]]).T

    # Perform the count for all relevant values of t
    out = tmp[np.argsort(np.hstack((tp, fp)))]
    out[:, 1:] = 1 - np.cumsum(out[:, 1:], axis=0) / np.array([fpN, tpN])
    
    # Filter out duplicate values of t
    dup = np.array([True] * (tpN + fpN))
    dup[:-1] = out[:-1, 0] != out[1:, 0]

    return out[dup, 1:]


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
    # Convert points to a numpy array
    points = np.array(points)
    
    # Sort points based on the x-coordinate
    tmp = np.vstack(([[0, 0]], points[np.lexsort(points.T)], [[1, 1]]))
    
    # Calculate the area under the ROC curve using trapezoidal rule
    auc = np.sum((tmp[1:, 0] - tmp[:-1, 0]) * (tmp[1:, 1] + tmp[:-1, 1]) / 2)
    
    return auc



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
    # Create a figure for the lift chart
    plt.figure(figsize=(9, 9))
    
    # Get the number of samples
    n = len(true_labels)
    
    # Create a temporary array to store the sample index and true labels
    tmp = np.zeros((n, 2))
    tmp[:, 0] = np.arange(1, n+1)
    tmp[:, 1] = np.array(true_labels)
    
    # Update the temporary array based on the positive class and predicted labels
    with np.nditer(tmp[:, 1], flags=['f_index'], op_flags=['readwrite']) as it:
        for x in it:
            if x[...] == pos and x[...] == predicted[it.index]:
                x[...] = 1
            else:
                x[...] = 0
    
    # Compute the cumulative sum of true positives
    tmp[:, 1] = np.cumsum(tmp[:, 1])
    
    # Plot the lift chart
    plt.scatter(tmp[:, 0], tmp[:, 1])
    plt.plot(tmp[:, 0], tmp[:, 1])
    
    # Set the x-axis label and ticks
    plt.xlabel('Sample Size', fontsize=15)
    plt.xticks(np.arange(1, n+1))
    
    # Set the y-axis label and ticks
    plt.ylabel('True Positives', fontsize=15)
    plt.yticks(np.arange(0, tmp[-1, 1]+1))
    
    # Set the title of the lift chart
    plt.title('Lift Chart', fontsize=15)
    
    # Display the lift chart
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
