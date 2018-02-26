import numpy as np


def recoverData(Z, U, K):
    """
    recovers an approximation the
    original data that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.
    """
    return (U[:, :K]@Z.T).T
