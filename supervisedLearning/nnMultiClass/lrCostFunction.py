import numpy as np
from regr import costFunctionReg
from regr import sigmoid


def lrCostFunction(theta, X, y, Lambda):
    """ computes the cost of using
        theta as the parameter for  logistic regression and the
        gradient of the cost w.r.t. to the parameters.
    """
    y = y[:, 0]
    m = y.shape
    hx = sigmoid(theta.dot(X.T))
    J = np.sum(-1*y*np.log(hx) - (-1*y + 1)*np.log(1 - hx))
    J /= m
    return J
