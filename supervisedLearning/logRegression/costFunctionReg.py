import numpy as np
from costFunction import costFunction


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
