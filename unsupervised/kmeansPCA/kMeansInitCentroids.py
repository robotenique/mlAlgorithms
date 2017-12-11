import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """
    return np.random.permutation(X)[:K]
