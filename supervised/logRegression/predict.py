import numpy as np
from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    return np.array([1 if theta.dot(xi) >= 0.5 else 0 for xi in X])
