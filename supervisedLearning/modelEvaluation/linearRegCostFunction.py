import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    grad = np.zeros(theta.shape)
    err = np.sum(theta*X, axis=1) - y # Matrix h(X) - y
    J = np.sum((err)**2)/(2*m) + Lambda*theta[1:]**2/(2*m)
    grad = np.sum((err.reshape(err.size, 1)*X).T, axis=1)/m + theta.T*Lambda/m
    grad[0] -= theta[0]*Lambda/m
    return J, grad
