import numpy as np

from predict import sigmoid


def predictOneVsAll(all_theta, X):
    """ will return a vector of predictions
        for each example in the matrix X. Note that X contains the examples in
        rows. all_theta is a matrix where the i-th row is a trained logistic
        regression theta vector for the i-th class.
    """

    m = X.shape[0]

    p = np.zeros((m, 1))
    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))
    class_prediction = (all_theta @ X.T).T

    for i in range(m):
        p[i] = np.argmax(class_prediction[i])

    return p    # add 1 to offset index of maximum in A row
