import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def validationCurve(X, y, Xval, yval):
    """returns the train
    and validation errors (in error_train, error_val)
    for different values of lambda. You are given the training set (X,
    y) and validation set (Xval, yval).
    """

    # Selected values of lambda (you should not change this)
    # lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    lambda_vec = np.linspace(0, 10, 1000)

    # You need to return these variables correctly.
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)
    lambdamin = 0
    min_err = np.inf
    for i in range(len(lambda_vec)):
        Lambda = lambda_vec[i]
        theta = trainLinearReg(X, y, Lambda)
        error_train[i], _ = linearRegCostFunction(X, y, theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)
        if(error_val[i] < min_err):
            min_err = error_val[i]
            lambdamin = Lambda

    return lambda_vec, error_train, error_val
