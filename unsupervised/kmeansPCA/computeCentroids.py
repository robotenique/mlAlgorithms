import numpy as np


def computeCentroids(X, idx, K):
    """
    returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = []
    for c in range(K):        
        clist = [X[p] for p, i in enumerate(idx) if i == c]
        centroids.append(np.sum(clist, axis=0)/len(clist))

    return centroids
