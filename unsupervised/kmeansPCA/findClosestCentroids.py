import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

    # Set K
    K = len(centroids)

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)
    dist = lambda x, y : np.sqrt(np.sum((x - y)**2))
    for p_idx, p in enumerate(X):
        minD = np.inf
        maxIdx = -1
        for i, c in enumerate(centroids):
            d = dist(p, c)
            if minD > d:
                minD = d
                maxIdx = i
        idx[p_idx] = maxIdx

    return idx
