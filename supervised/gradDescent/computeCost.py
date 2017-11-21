import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0
    hx = lambda xRow, omegaVec: omegaVec.dot(xRow)
    for xRow, yRow in zip(X, y):
        J += (hx(xRow, theta) - yRow)**2
    J /= 2*m

    return J
