import numpy as np
import math as m

def innerG(theta, x, y, jth, Lambda,  mul):
    ret = np.sum([(sigmoid(theta.dot(xi)) - yi)* xi[jth] for xi, yi in zip(x, y)])
    return ret + mul*Lambda*theta[jth]


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """    
    y = np.squeeze(y)
    m = y.shape # number of training samples
    grad = X.T.dot(sigmoid(theta.dot(X.T))-1*y)
    grad[1:] = grad[1:] + Lambda*theta[1:]
    grad /= m

    return grad

def innerA(t, xi, yi, jth):
    return (sigmoid(t.dot(xi)) - yi) * xi[jth]


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression
    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    y = y[:, 0]
    m = y.shape # number of training samples
    grad = X.T.dot(sigmoid(theta.dot(X.T))-1*y)
    grad /= m
    return grad


def inner(t, xi, yi):
    return -yi*np.log(sigmoid(t.dot(xi))) - (1 - yi)*np.log(1 - sigmoid(t.dot(xi)))

def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # Initialize some useful values
    m = y.size  # number of training examples
    J = np.sum(np.array([inner(theta, xi, yi) for xi, yi in zip(X, y)]))
    J /= m


    return J

def sigmoid(z):
    """computes the sigmoid of z."""
    g = (1 + np.exp(-z))**-1
    return g


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples
    j = costFunction(theta, X, y)
    j += (Lambda/(2*m))*np.sum(theta[1:]**2)
    return j
