import numpy as np


def pca(X):
    """computes eigenvectors of the covariance matrix of X
      Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # Useful values
    m, n = X.shape
    sigma = X.T.dot(X) / float(m)
    U, S, V = np.linalg.svd(sigma)    
    return U, np.diag(S), V
