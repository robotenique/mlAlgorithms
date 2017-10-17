import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def learningCurve(X, y, Xval, yval, Lambda):
    """returns the train and
    cross validation set errors for a learning curve. In particular,
    it returns two vectors of the same length - error_train and
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
    """

    # Number of training examples
    m, _ = X.shape

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        theta = trainLinearReg(X[:i + 1], y[:i + 1], Lambda)
        error_train[i], _ = linearRegCostFunction(X[:i + 1], y[:i + 1], theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)
    
    return  error_train, error_val
