import numpy as np
from sigmoid import sigmoid

def inner(t, xi, yi, jth):
    return (sigmoid(t.dot(xi)) - yi) * xi[jth]


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression
    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = y.size # number of training samples
    n = X[0].size # number of features
    grad = np.zeros(n)
    for j in range(n):
        grad[j] = 1/m * np.sum(np.array([inner(theta, xi, yi, j) for xi, yi in zip(X, y)]))

    return grad
