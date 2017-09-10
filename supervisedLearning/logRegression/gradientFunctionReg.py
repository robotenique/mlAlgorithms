import numpy as np
from gradientFunction import gradientFunction
from sigmoid import sigmoid

def innerG(theta, x, y, jth, Lambda,  mul):
    ret = np.sum([(sigmoid(theta.dot(xi)) - yi)* xi[jth] for xi, yi in zip(x, y)])
    return ret + mul*Lambda*theta[jth]


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples
    grad = np.array([innerG(theta, X, y, j, Lambda, 1 if j != 0 else 0) for j in range(theta.size)])
    return 1/m * grad
