import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    n = X[0].size
    m = X.T[0].size
    X_norm = np.copy(X)
    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    norm_f = lambda x, n: (x - mu[n])/sigma[n]
    for i in range(m):
        for j in range(n):
            X_norm[i][j] = norm_f(X_norm[i][j], j)
    return X_norm, mu, sigma

def normEntry(Xi, mu, sigma):
    """
        Normalizes a single entry Xi, provided the mu and sigma matrices.
        No size check is made; Xi is a LIST, not an numpy array.
    """
    singNorm = lambda x, i: (x - mu[i - 1])/sigma[i - 1]
    # Don't count the first column
    for i in range(1, len(Xi)):
        Xi[i] = singNorm(Xi[i], i)
    return Xi
