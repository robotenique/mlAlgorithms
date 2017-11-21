import numpy as np
from scipy.optimize import minimize
from lrCostFunction import lrCostFunction
from mlCalcs import gradientFunctionReg


def optimize(Lambda, X, y, initial_theta):
    result = minimize(lrCostFunction, initial_theta, method='L-BFGS-B',
                      jac=gradientFunctionReg, args=(X, y, Lambda),
                      options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    return result


def oneVsAll(X, y, num_labels, Lambda):
    """ trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

    # Some useful variables
    m, n = X.shape

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))
    y = np.squeeze(y) # Collapse y to 1D
    all_theta = np.empty((num_labels, n + 1))
    # Set Initial theta
    initial_theta = np.zeros((n + 1, 1))
    for c in range(num_labels):
        pos = c
        if c == 0:
            c = 10
            pos = 0
        yVals = np.where(y == c, 1, 0)
        all_theta[pos] = optimize(Lambda, X, yVals, initial_theta).x

    return all_theta
