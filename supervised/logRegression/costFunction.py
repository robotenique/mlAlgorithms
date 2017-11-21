import math as m
import numpy as np
from sigmoid import sigmoid

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
