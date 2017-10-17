import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    grad = np.zeros(theta.shape)
    int_err = np.sum(theta*X, axis=1) - y # h(x) - y
    J = np.sum((int_err)**2)/(2*m) + Lambda*theta[1:]**2/(2*m)
    print(np.sum(int_err))
    print(theta.T[0]*np.sum(int_err)/m)
    grad[0] = theta.T[0]*np.sum(int_err)/m
    1*-183.63
    grad[1] = (theta.T[1:]*np.sum(int_err))/m + theta.T[1:]*Lambda/m
    print(grad[1])
    exit()
    return J, grad
